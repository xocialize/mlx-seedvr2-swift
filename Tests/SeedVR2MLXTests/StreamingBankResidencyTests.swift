// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Budgeted bank residency (`GAP-PROGRAM.md` V12-B). The properties that matter are (a) the default
// changes nothing, (b) under a budget the resident bytes actually stay under it, and (c) what comes
// back out is what went in — because these banks ARE the causal state, and a residency that quietly
// corrupted one would show up as an intermittent chunk-boundary pop on large outputs only.
import Foundation
import MLX
import MLXRandom
import XCTest

@testable import SeedVR2MLX

final class StreamingBankResidencyTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    /// Distinct, identifiable banks — value equality is what the round-trip assertions check.
    private func bank(_ id: Int, elements: Int = 4096) -> VAEStreamingBank {
        let a = MLXRandom.uniform(low: -1, high: 1, [1, 1, 1, elements / 64, 64],
                                  key: MLXRandom.key(UInt64(500 + id))).asType(VAEPrecision.dtype)
        return VAEStreamingBank(tails: [a])
    }

    private func maxDelta(_ a: VAEStreamingBank, _ b: VAEStreamingBank) -> Float {
        var worst: Float = 0
        for (x, y) in zip(a.tails, b.tails) {
            guard let x, let y else { continue }
            worst = max(worst, (x.asType(.float32) - y.asType(.float32)).abs().max().item(Float.self))
        }
        return worst
    }

    /// 🔑 **Default is no eviction.** Nothing is written, everything stays resident — the pre-existing
    /// behaviour, so adopting this type cannot change an existing run.
    func testDefaultNeverEvicts() throws {
        let residency = try StreamingBankResidency()
        for i in 0 ..< 20 { try residency.store(bank(i), at: i) }
        XCTAssertEqual(residency.spilledCount, 0)
        XCTAssertEqual(residency.statistics.spills, 0)
        XCTAssertEqual(residency.statistics.bytesSpilled, 0)
        for i in 0 ..< 20 {
            XCTAssertNotNil(try residency.take(at: i))
        }
        XCTAssertEqual(residency.statistics.loads, 0, "nothing should have come from disk")
    }

    /// 🔑 **Under a budget, resident bytes stay under it** — the whole point. Checked after every
    /// store, not just at the end, because a transient overshoot is exactly what an OOM would catch.
    func testResidentBytesNeverExceedBudget() throws {
        let one = bank(0).byteCount
        let budget = one * 3
        let residency = try StreamingBankResidency(budgetBytes: budget)
        for i in 0 ..< 25 {
            try residency.store(bank(i), at: i)
            XCTAssertLessThanOrEqual(residency.residentBytes, budget,
                                     "resident bytes exceeded the budget after storing \(i)")
        }
        XCTAssertGreaterThan(residency.statistics.spills, 0, "nothing spilled — budget never bound")
    }

    /// 🔑 **A spilled bank comes back bit-identical.** The store's own round-trip test covers the file
    /// format; this covers the residency's bookkeeping — right index, right file, nothing crossed.
    func testSpilledBanksReturnBitIdentical() throws {
        let one = bank(0).byteCount
        let residency = try StreamingBankResidency(budgetBytes: one * 2)
        var originals: [Int: VAEStreamingBank] = [:]
        for i in 0 ..< 12 {
            let b = bank(i)
            originals[i] = b
            try residency.store(b, at: i)
        }
        XCTAssertGreaterThan(residency.spilledCount, 0)
        for i in 0 ..< 12 {
            let restored = try XCTUnwrap(try residency.take(at: i), "bank \(i) vanished")
            XCTAssertEqual(maxDelta(originals[i]!, restored), 0.0,
                           "bank \(i) changed across eviction")
        }
    }

    /// ⚠️ A budget smaller than a single bank — including 0 — must still work, by spilling everything.
    /// The alternative (silently exceeding the budget) is the failure mode this class exists to stop.
    func testBudgetSmallerThanOneBankSpillsEverything() throws {
        let residency = try StreamingBankResidency(budgetBytes: 0)
        for i in 0 ..< 5 { try residency.store(bank(i), at: i) }
        XCTAssertEqual(residency.residentBytes, 0)
        XCTAssertEqual(residency.spilledCount, 5)
        for i in 0 ..< 5 {
            XCTAssertEqual(maxDelta(bank(i), try XCTUnwrap(try residency.take(at: i))), 0.0)
        }
        XCTAssertEqual(residency.statistics.hits, 0, "budget 0 cannot serve anything from memory")
        XCTAssertEqual(residency.statistics.loads, 5)
    }

    /// Storing `nil` clears a position — the driver's `endsSegment` path — and must remove any spill
    /// file too, or a later chunk would resurrect stale causal state.
    func testStoringNilClearsResidentAndSpilled() throws {
        let residency = try StreamingBankResidency(budgetBytes: 0)
        try residency.store(bank(1), at: 1)
        XCTAssertEqual(residency.spilledCount, 1)
        try residency.store(nil, at: 1)
        XCTAssertEqual(residency.spilledCount, 0)
        XCTAssertNil(try residency.take(at: 1), "cleared position must read back as no state")
    }

    /// `reset()` is the flush path (scene cut, clip end): everything gone, no files left behind.
    func testResetClearsEverything() throws {
        let residency = try StreamingBankResidency(budgetBytes: bank(0).byteCount)
        for i in 0 ..< 8 { try residency.store(bank(i), at: i) }
        residency.reset()
        XCTAssertEqual(residency.residentBytes, 0)
        XCTAssertEqual(residency.spilledCount, 0)
        for i in 0 ..< 8 { XCTAssertNil(try residency.take(at: i)) }
    }

    /// Re-storing an index replaces it rather than accumulating — the steady-state path, where every
    /// tile position is written once per chunk. A leak here would grow without bound over a long clip.
    func testRestoringSameIndexDoesNotAccumulate() throws {
        let residency = try StreamingBankResidency()
        for _ in 0 ..< 50 { try residency.store(bank(7), at: 7) }
        XCTAssertEqual(residency.residentBytes, bank(7).byteCount,
                       "resident accounting grew across repeated stores to one index")
    }

    /// Under the driver's strictly cyclic access order, a K-bank budget yields exactly K hits per
    /// cycle of N. Pinned because it is the basis of the cost model: hits are free, misses cost a
    /// read, and the ratio is what makes the overhead predictable rather than incidental.
    func testCyclicAccessYieldsBudgetSizedHitsPerCycle() throws {
        let one = bank(0).byteCount
        let capacity = 3, positions = 10
        let residency = try StreamingBankResidency(budgetBytes: one * capacity)
        for i in 0 ..< positions { try residency.store(bank(i), at: i) }   // cycle 1
        let before = residency.statistics
        for i in 0 ..< positions {                                          // cycle 2
            _ = try residency.take(at: i)
            try residency.store(bank(i), at: i)
        }
        let hits = residency.statistics.hits - before.hits
        XCTAssertEqual(hits, capacity,
                       "expected exactly \(capacity) hits per cycle of \(positions), got \(hits)")
    }
}
