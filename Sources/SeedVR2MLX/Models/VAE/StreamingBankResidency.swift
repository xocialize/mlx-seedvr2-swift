// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Budgeted residency for per-tile streaming banks — the second half of bank offload
// (`GAP-PROGRAM.md` V12-B). `VAEStreamingBankStore` gave a bank the ability to leave memory; this
// decides WHICH banks leave and when.
//
// 🔑 **The problem, in one line:** the tiled temporal driver keeps one bank per spatial tile position,
// ~0.765 GiB each, all live between uses — so bank residency is `positions × 0.765 GiB`: 6.9 GiB at a
// 512² output, **26.8 GiB at SD ×2**, 130 GiB at HD→4K. On a 16 GB machine the SD case exceeds the
// whole machine, and neither tile size nor the temporal window moves it (both measured dead, and pinned
// in `VAEStreamingBankSizeTests`).
//
// 🔑 **What it costs, MEASURED end to end** (`probes/v12b2_bank_eviction.out`, 17 frames at 40 tile
// positions through the shipping surface, arm A bracketed before and after arm B):
//
//     peak footprint  42.02 GiB → 12.69 GiB   (3.31×, and 16 GB becomes reachable)
//     wall clock      227 s mean → 335.6 s    (**+48%**, at a 1 GiB budget = 39 of 40 banks spilling)
//     output          max |Δ| = 0 over 18 frames — bit-identical
//
// 🚨 **That +48% is the THIRD cost estimate for this work, and the first one measured. The two before
// it were wrong in opposite directions, both for the same reason.** V12-B said "2.7×, offload cannot
// work" by comparing I/O at 170 tile positions against compute measured at 9. The pre-measurement
// estimate here said "~6–13%" by dividing bank bytes by RAW DISK BANDWIDTH. **A ratio of two
// quantities measured separately, under different conditions, is not a measurement — measure the
// composite.**
//
// 🔑 **Where the 48% goes, and why it is headroom rather than a floor:** 59.7 GiB written + 59.7 GiB
// read in 108.6 s is ~1.27 GB/s effective, against 8.74 GB/s fsync'd write measured on the same
// volume. **The round trip is SERIALIZATION-bound, not bandwidth-bound** — 57 separate tensors per
// bank through safetensors framing plus host materialization per array. A contiguous single-buffer
// bank format is the obvious first move if this ever blocks anything. Not attempted.
//
// ⚠️ 1 GiB is the EXTREME budget. A host should set the largest budget that fits, not the smallest
// that works; overhead falls as fewer banks spill.
//
// ⚠️ **The real cost of eviction is WRITE VOLUME, not latency.** A 10 s 24 fps clip at SD ×2 spills
// roughly 800 GiB through the scratch volume. That is why eviction is **budget-driven and off by
// default**: a machine with room keeps every bank resident and writes nothing, and only a machine that
// would otherwise be unable to run the job at all pays the SSD wear. Never enable it as a blanket
// optimisation.
//
// ⚠️ **Eviction only moves `phys_footprint` because the engine caps the MLX buffer cache.** Dropping the
// last reference returns a tail's buffer to MLX's pool, not to the OS; it is reclaimed only once the
// pool is over its cap. `PORT-QUEUE.md` cross-cutting note 3 — *set `GPU.set(cacheLimit:)`, never
// `memoryLimit`* — is what makes this real rather than bookkeeping.
import Foundation
import MLX

/// Holds one streaming bank per tile position, keeping resident only what fits a byte budget and
/// spilling the rest to a scratch directory.
///
/// **Not thread-safe by design** — the driver walks tile positions sequentially inside one chunk, and
/// adding a lock here would suggest a concurrency this class has never been exercised under.
public final class StreamingBankResidency {

    /// What happened, for receipts and for deciding whether a budget was set sensibly.
    public struct Statistics: Sendable, Equatable {
        /// `take` served from memory.
        public var hits = 0
        /// `take` had to read from disk.
        public var loads = 0
        /// `store` wrote a bank out.
        public var spills = 0
        /// Total bytes written across the run — the SSD-wear number, which is the cost that matters.
        public var bytesSpilled = 0
    }

    /// Byte budget for RESIDENT banks. `nil` disables eviction entirely: every bank stays in memory and
    /// nothing is ever written, which is the pre-existing behaviour and remains the default.
    ///
    /// 🚨 **A budget of 0 is not the same as `nil`.** Zero means "evict everything", which is legal and
    /// costs a round-trip per bank per chunk. `nil` means "never evict".
    public let budgetBytes: Int?

    private let directory: URL
    private let ownsDirectory: Bool
    private var resident: [Int: VAEStreamingBank] = [:]
    private var spilled: Set<Int> = []
    private var residentByteCount = 0
    private var stats = Statistics()

    /// - Parameters:
    ///   - budgetBytes: resident-bank budget. `nil` (default) never evicts.
    ///   - directory: where spill files go. `nil` creates a temporary directory owned by this
    ///     instance and removed on `reset()`/`deinit`. Pass a location on fast local storage; a
    ///     network volume would make the round-trip dominate.
    public init(budgetBytes: Int? = nil, directory: URL? = nil) throws {
        self.budgetBytes = budgetBytes
        if let directory {
            self.directory = directory
            self.ownsDirectory = false
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } else {
            self.directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("seedvr2-banks-\(UUID().uuidString)")
            self.ownsDirectory = true
            try FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true)
        }
    }

    deinit { if ownsDirectory { try? FileManager.default.removeItem(at: directory) } }

    // MARK: - Surface

    public var statistics: Statistics { stats }
    public var residentBytes: Int { residentByteCount }
    public var spilledCount: Int { spilled.count }

    private func url(_ index: Int) -> URL {
        directory.appendingPathComponent("bank-\(index).safetensors")
    }

    /// Hand a tile position's bank to the residency. `nil` clears that position.
    ///
    /// ⚠️ Storing evaluates and possibly writes, so this is where a chunk's cost lands. It is called
    /// once per tile position per chunk by the driver.
    public func store(_ bank: VAEStreamingBank?, at index: Int) throws {
        drop(index)
        guard let bank else { return }

        guard let budget = budgetBytes else {
            insertResident(bank, at: index)
            return
        }
        // 🚨 **Policy: keep a STABLE resident set — spill the incoming bank rather than evicting a
        // sitting one.** This looks lazier than LRU and is strictly better here, for a reason specific
        // to how the driver walks tiles.
        //
        // Access is exactly cyclic: 0, 1, …, N−1, 0, 1, … Under that pattern **LRU (and plain
        // insertion-order eviction) is PESSIMAL — it yields ZERO hits**, because the oldest resident is
        // always the one needed soonest, so every store evicts precisely the bank the next cycle is
        // about to ask for. An earlier revision of this file did evict oldest-first and
        // `testCyclicAccessYieldsBudgetSizedHitsPerCycle` caught it at 0 hits instead of `capacity`.
        //
        // Keeping whichever banks arrived first pins a fixed working set, so the low-numbered tile
        // positions are served from memory every cycle: **exactly `budget / bankBytes` hits per cycle
        // of N**, which is what makes the cost model predictable rather than incidental.
        let incoming = bank.byteCount
        if residentByteCount + incoming <= budget {
            insertResident(bank, at: index)
        } else {
            try write(bank, at: index)
        }
    }

    /// Retrieve a tile position's bank, loading it back if it was spilled. Returns `nil` if that
    /// position has no state — a cold stream, which is a normal condition, not an error.
    public func take(at index: Int) throws -> VAEStreamingBank? {
        if let bank = resident[index] {
            stats.hits += 1
            return bank
        }
        guard spilled.contains(index) else { return nil }
        let bank = try VAEStreamingBank.read(from: url(index))
        stats.loads += 1
        return bank
    }

    /// Drop all state and remove every spill file — the stream-flush path (scene cut, clip end).
    public func reset() {
        resident.removeAll()
        residentByteCount = 0
        for index in spilled { try? FileManager.default.removeItem(at: url(index)) }
        spilled.removeAll()
    }

    // MARK: - Internals

    private func insertResident(_ bank: VAEStreamingBank, at index: Int) {
        resident[index] = bank
        residentByteCount += bank.byteCount
    }

    /// Forget whatever is held for `index`, in memory or on disk.
    private func drop(_ index: Int) {
        if let existing = resident.removeValue(forKey: index) {
            residentByteCount -= existing.byteCount
        }
        if spilled.remove(index) != nil {
            try? FileManager.default.removeItem(at: url(index))
        }
    }

    private func write(_ bank: VAEStreamingBank, at index: Int) throws {
        try bank.write(to: url(index))
        spilled.insert(index)
        stats.spills += 1
        stats.bytesSpilled += bank.byteCount
    }
}
