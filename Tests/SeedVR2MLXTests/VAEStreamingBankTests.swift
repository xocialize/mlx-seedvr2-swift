// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Streaming bank export/adopt (V12-D): one VAE, several interleaved temporal streams — the
// tiled temporal driver's per-tile-position state. Weightless in the sense of needing no
// checkpoint: parameters are randomized so value comparisons are non-vacuous.
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import SeedVR2MLX

final class VAEStreamingBankTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    /// A VAE with deterministic random parameters (zeros would make every value assertion
    /// vacuous — conv outputs would be identically zero regardless of state).
    private func makeRandomizedVAE() throws -> SeedVR2VAE {
        let vae = SeedVR2VAE()
        var params: [(String, MLXArray)] = []
        for (i, (key, value)) in vae.parameters().flattened().enumerated() {
            params.append((key, MLXRandom.uniform(low: -0.05, high: 0.05, value.shape,
                                                  key: MLXRandom.key(UInt64(1000 + i)))))
        }
        try vae.update(parameters: ModuleParameters.unflattened(params), verify: .none)
        return vae
    }

    /// Two temporal streams interleaved through ONE VAE with per-stream banks are bit-identical
    /// to each stream run alone — the property the tiled temporal driver rests on (tile A's
    /// state must never condition tile B's chunk).
    func testInterleavedStreamsWithBanksMatchSequentialRuns() throws {
        let vae = try makeRandomizedVAE()
        func clip(_ key: UInt64) -> [MLXArray] {
            let whole = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, 13, 32, 32],
                                          key: MLXRandom.key(key))
            return [whole[0..., 0..., 0 ..< 5], whole[0..., 0..., 5 ..< 9],
                    whole[0..., 0..., 9 ..< 13]]
        }
        let (clipA, clipB) = (clip(7), clip(8))

        // Interleaved: A0 B0 A1 B1 A2 B2 through one VAE, swapping banks per stream.
        var bankA: VAEStreamingBank?, bankB: VAEStreamingBank?
        var interA: [MLXArray] = [], interB: [MLXArray] = []
        for i in 0 ..< 3 {
            let state: VAEMemoryState = i == 0 ? .initializing : .active
            vae.adoptStreamingMemory(bankA)
            let a = vae.decode(vae.encode(clipA[i], memoryState: state), memoryState: state)
            eval([a] + vae.streamingMemoryTails())
            bankA = vae.exportStreamingMemory()
            interA.append(a)

            vae.adoptStreamingMemory(bankB)
            let b = vae.decode(vae.encode(clipB[i], memoryState: state), memoryState: state)
            eval([b] + vae.streamingMemoryTails())
            bankB = vae.exportStreamingMemory()
            interB.append(b)
        }

        // Each stream alone through the same VAE (reset between), the plain V12-S shape.
        vae.resetStreamingMemory()
        var seqA: [MLXArray] = []
        for (i, chunk) in clipA.enumerated() {
            let state: VAEMemoryState = i == 0 ? .initializing : .active
            let out = vae.decode(vae.encode(chunk, memoryState: state), memoryState: state)
            eval([out] + vae.streamingMemoryTails())
            seqA.append(out)
        }
        vae.resetStreamingMemory()
        var seqB: [MLXArray] = []
        for (i, chunk) in clipB.enumerated() {
            let state: VAEMemoryState = i == 0 ? .initializing : .active
            let out = vae.decode(vae.encode(chunk, memoryState: state), memoryState: state)
            eval([out] + vae.streamingMemoryTails())
            seqB.append(out)
        }
        vae.resetStreamingMemory()

        for i in 0 ..< 3 {
            XCTAssertEqual(abs(interA[i] - seqA[i]).max().item(Float.self), 0,
                           "stream A chunk \(i): interleaved-with-banks ≠ sequential")
            XCTAssertEqual(abs(interB[i] - seqB[i]).max().item(Float.self), 0,
                           "stream B chunk \(i): interleaved-with-banks ≠ sequential")
        }

        // Non-vacuity: WITHOUT bank isolation the interleave contaminates — B's tail feeding
        // A's `.active` chunk changes the output. If this ever stops differing, the value
        // assertions above stopped proving anything.
        vae.resetStreamingMemory()
        var naive: [MLXArray] = []
        for i in 0 ..< 3 {
            let state: VAEMemoryState = i == 0 ? .initializing : .active
            let a = vae.decode(vae.encode(clipA[i], memoryState: state), memoryState: state)
            eval([a] + vae.streamingMemoryTails())
            naive.append(a)
            let b = vae.decode(vae.encode(clipB[i], memoryState: state), memoryState: state)
            eval([b] + vae.streamingMemoryTails())
        }
        vae.resetStreamingMemory()
        XCTAssertGreaterThan(abs(naive[1] - seqA[1]).max().item(Float.self), 0,
                             "cross-stream contamination must be measurable without banks")
    }

    /// `adoptStreamingMemory(nil)` is `resetStreamingMemory()`; an exported bank survives a
    /// reset and restores the exact tails.
    func testAdoptNilResetsAndBankRestores() throws {
        let vae = try makeRandomizedVAE()
        let x = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, 5, 32, 32], key: MLXRandom.key(3))
        let z = vae.encode(x, memoryState: .initializing)
        eval([vae.decode(z, memoryState: .initializing)] + vae.streamingMemoryTails())
        XCTAssertFalse(vae.streamingMemoryTails().isEmpty)

        let bank = vae.exportStreamingMemory()
        let tailCount = vae.streamingMemoryTails().count
        vae.adoptStreamingMemory(nil)
        XCTAssertTrue(vae.streamingMemoryTails().isEmpty, "adopt(nil) must clear every tail")
        vae.adoptStreamingMemory(bank)
        XCTAssertEqual(vae.streamingMemoryTails().count, tailCount, "bank restores all tails")
    }
}
