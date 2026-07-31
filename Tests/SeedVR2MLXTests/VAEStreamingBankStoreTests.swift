// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Round-trip fidelity for bank offload (`GAP-PROGRAM.md` V12-B). The contract is exact: a bank that
// leaves memory and comes back must be BIT-IDENTICAL, because it is the causal state that makes chunk
// N+1 continue chunk N. Anything less reintroduces the chunk-boundary pop that V12-S closed
// (1.4089 → 0.3426 max |ΔL*|), and it would do so intermittently, only on the outputs large enough to
// trigger eviction — the worst possible failure shape.
//
// Weightless: bank shapes and values come from a randomized VAE, so this needs no checkpoint.
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import SeedVR2MLX

final class VAEStreamingBankStoreTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("bank-\(UUID().uuidString).safetensors")
    }

    /// A bank of known tails, built directly — no VAE needed to exercise the store itself.
    private func makeBank(count: Int, nilAt: Set<Int> = []) -> VAEStreamingBank {
        var tails = [MLXArray?](repeating: nil, count: count)
        for i in 0 ..< count where !nilAt.contains(i) {
            tails[i] = MLXRandom.uniform(low: -1, high: 1, [1, 2 + i % 3, 2, 4, 4],
                                         key: MLXRandom.key(UInt64(90 + i)))
                .asType(VAEPrecision.dtype)
        }
        return VAEStreamingBank(tails: tails)
    }

    /// 🔑 The core contract: written and read back, every tail is bit-identical.
    func testRoundTripIsBitIdentical() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let original = makeBank(count: 12)

        try original.write(to: url)
        let restored = try VAEStreamingBank.read(from: url)

        XCTAssertEqual(restored.tails.count, original.tails.count)
        XCTAssertEqual(restored.byteCount, original.byteCount)
        for (i, (a, b)) in zip(original.tails, restored.tails).enumerated() {
            guard let a, let b else { return XCTFail("tail \(i) nil-ness changed") }
            XCTAssertEqual(a.dtype, b.dtype, "tail \(i) dtype changed — a widened tail is not free")
            XCTAssertEqual(a.shape, b.shape, "tail \(i) shape changed")
            let maxDelta = (a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self)
            XCTAssertEqual(maxDelta, 0.0, "tail \(i) changed value — streaming state must survive exactly")
        }
    }

    /// ⚠️ `nil` slots must survive, including trailing ones. `adoptStreamingMemory` installs tails
    /// POSITIONALLY and preconditions on the count, so a bank that lost its trailing nils would trip
    /// that precondition and one that collapsed interior nils would install every subsequent tail onto
    /// the wrong convolution — silently, and only in the eviction path.
    func testNilSlotsSurviveIncludingTrailing() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let original = makeBank(count: 10, nilAt: [0, 4, 5, 9])

        try original.write(to: url)
        let restored = try VAEStreamingBank.read(from: url)

        XCTAssertEqual(restored.tails.count, 10)
        for i in 0 ..< 10 {
            XCTAssertEqual(restored.tails[i] == nil, original.tails[i] == nil,
                           "nil-ness of slot \(i) changed")
        }
    }

    /// An all-nil bank (a freshly reset stream) round-trips rather than failing.
    func testEmptyBankRoundTrips() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let original = makeBank(count: 6, nilAt: Set(0 ..< 6))
        try original.write(to: url)
        let restored = try VAEStreamingBank.read(from: url)
        XCTAssertEqual(restored.tails.count, 6)
        XCTAssertEqual(restored.byteCount, 0)
        XCTAssertTrue(restored.tails.allSatisfy { $0 == nil })
    }

    /// A file that is not a bank must be REFUSED, not silently adopted. Positional installation means
    /// a foreign file would land arbitrary arrays on arbitrary convolutions.
    func testForeignFileIsRejected() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: ["something": MLXArray([1, 2, 3])], url: url)
        XCTAssertThrowsError(try VAEStreamingBank.read(from: url)) { error in
            guard case VAEStreamingBankStoreError.malformed = error else {
                return XCTFail("expected .malformed, got \(error)")
            }
        }
    }

    /// 🔑 The end-to-end property the driver actually depends on: a bank exported from a real VAE,
    /// evicted to disk, restored and adopted, continues the stream **exactly** as the in-memory bank
    /// would have. This is the assertion that makes offload safe to turn on — the store round-tripping
    /// its own bytes (above) is necessary but does not prove the VAE agrees.
    func testEvictedBankContinuesTheStreamIdentically() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let vae = SeedVR2VAE()
        var params: [(String, MLXArray)] = []
        for (i, (key, value)) in vae.parameters().flattened().enumerated() {
            params.append((key, MLXRandom.uniform(low: -0.05, high: 0.05, value.shape,
                                                  key: MLXRandom.key(UInt64(4000 + i)))
                                    .asType(VAEPrecision.dtype)))
        }
        try vae.update(parameters: ModuleParameters.unflattened(params), verify: .none)

        func chunk(_ key: UInt64, frames: Int) -> MLXArray {
            MLXRandom.uniform(low: -1, high: 1, [1, 3, frames, 32, 32], key: MLXRandom.key(key))
                .asType(VAEPrecision.dtype)
        }
        let first = chunk(21, frames: 5), second = chunk(22, frames: 4)

        // Run chunk 1, export the bank.
        vae.resetStreamingMemory()
        _ = vae.decode(vae.encode(first, memoryState: .initializing), memoryState: .initializing)
        let bank = vae.exportStreamingMemory()

        // Arm A — carry the bank in memory (what the driver does today).
        vae.adoptStreamingMemory(bank)
        let inMemory = vae.decode(vae.encode(second, memoryState: .active), memoryState: .active)
        eval(inMemory)

        // Arm B — evict to disk, drop it, restore, adopt.
        try bank.write(to: url)
        let restored = try VAEStreamingBank.read(from: url)
        vae.adoptStreamingMemory(restored)
        let evicted = vae.decode(vae.encode(second, memoryState: .active), memoryState: .active)
        eval(evicted)

        XCTAssertEqual(inMemory.shape, evicted.shape)
        let maxDelta = (inMemory.asType(.float32) - evicted.asType(.float32)).abs().max().item(Float.self)
        XCTAssertEqual(maxDelta, 0.0,
                       "an evicted bank must continue the stream bit-identically; max|Δ| = \(maxDelta)")

        // Non-vacuity: the continuation must actually depend on the bank, or the assertion above
        // would pass for a store that returned garbage. A cold stream must differ.
        vae.resetStreamingMemory()
        let cold = vae.decode(vae.encode(second, memoryState: .active), memoryState: .active)
        eval(cold)
        let coldDelta = (inMemory.asType(.float32) - cold.asType(.float32)).abs().max().item(Float.self)
        XCTAssertGreaterThan(coldDelta, 0.0,
                             "carrying state must change the output, else this test proves nothing")
    }
}
