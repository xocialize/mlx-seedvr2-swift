// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Causal temporal round trip: T frames in -> T frames out (V12-follow, 2026-07-30).
//
// The encoder compresses causally — latT = 1 + (T-1)/4 (first frame special, then 4x) — and
// the decoder must invert it: 4*(latT-1) + 1. The upstream mflux Upsample3D gated its causal
// `remove_head` on T == 1 (the only case an image pipeline has), so latT 2/3/4 decoded to
// 8/12/16 frames instead of 5/9/13. These tests pin the rule at the shape level; they run
// weightless (shapes don't depend on values), so they hold in CI without the model.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import SeedVR2MLX

final class VAETemporalRoundTripTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    /// Encoder honours the causal rule and the decoder inverts it, T ∈ {1, 5, 9, 13}.
    func testEncodeDecodeFrameCountRoundTrip() {
        let vae = SeedVR2VAE()
        for t in [1, 5, 9, 13] {
            let x = MLXArray.zeros([1, 3, t, 32, 32], type: Float.self)
            let z = vae.encode(x)
            eval(z)
            XCTAssertEqual(z.shape[2], 1 + (t - 1) / 4, "encode T=\(t): latT")
            let y = vae.decode(z)
            eval(y)
            XCTAssertEqual(y.shape[2], t, "decode T=\(t): frame count")
            // Spatial round trip: /8 encode then x8 decode returns the input dims.
            XCTAssertEqual(y.shape[3], x.shape[3], "decode T=\(t): spatial H")
            XCTAssertEqual(y.shape[4], x.shape[4], "decode T=\(t): spatial W")
        }
    }

    /// The decoder's causal inverse pinned directly, independent of the encoder:
    /// latT ∈ {1, 2, 3, 4} -> 4*(latT-1) + 1 frames.
    func testDecoderCausalInverse() {
        let vae = SeedVR2VAE()
        for latT in 1 ... 4 {
            let z = MLXArray.zeros([1, 16, latT, 4, 4], type: Float.self)
            let y = vae.decode(z)
            eval(y)
            XCTAssertEqual(y.shape[2], 4 * (latT - 1) + 1, "decode latT=\(latT)")
        }
    }

    /// The T=1 shipping path is unchanged: a 4D single frame and its explicit 5D form
    /// produce identical latents and identical decodes (the ndim==4 expansion seam).
    func testSingleFramePathUnchanged() {
        let vae = SeedVR2VAE()
        let x4 = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, 32, 32], key: MLXRandom.key(3))
        let x5 = x4.expandedDimensions(axis: 2)
        let (z4, z5) = (vae.encode(x4), vae.encode(x5))
        eval(z4, z5)
        XCTAssertEqual(z4.shape, z5.shape)
        XCTAssertEqual(abs(z4 - z5).max().item(Float.self), 0)
        let (y4, y5) = (vae.decode(z4), vae.decode(z5))
        eval(y4, y5)
        XCTAssertEqual(y4.shape[2], 1)
        XCTAssertEqual(abs(y4 - y5).max().item(Float.self), 0)
    }

    /// Streaming memory (V12-S): a clip cut into chunks round-trips to the same frame count as
    /// the same clip in one pass, and the causal arithmetic differs per chunk exactly as SeedVR's
    /// `slicing_encode` / `slicing_decode` require — first chunk 4k+1 frames -> latT k+1 -> 4k+1
    /// frames back, every later chunk 4k frames -> latT k -> 4k frames back (no `remove_head`,
    /// because only the first chunk carries the encoder's first-frame special case).
    func testStreamingChunkedRoundTrip() {
        let vae = SeedVR2VAE()
        for (first, rest) in [(5, 4), (9, 8), (13, 12)] {
            vae.resetStreamingMemory()
            var total = 0
            for (i, t) in [first, rest, rest].enumerated() {
                let state: VAEMemoryState = i == 0 ? .initializing : .active
                let x = MLXArray.zeros([1, 3, t, 32, 32], type: Float.self)
                let z = vae.encode(x, memoryState: state)
                eval(z)
                XCTAssertEqual(z.shape[2], i == 0 ? 1 + (t - 1) / 4 : t / 4,
                               "stream chunk \(i) (T=\(t)): latT")
                let y = vae.decode(z, memoryState: state)
                eval([y] + vae.streamingMemoryTails())
                XCTAssertEqual(y.shape[2], t, "stream chunk \(i) (T=\(t)): frames in == frames out")
                XCTAssertEqual(y.shape[3], 32); XCTAssertEqual(y.shape[4], 32)
                total += t
            }
            XCTAssertEqual(total, first + 2 * rest)
        }
    }

    /// Streaming actually carries state: with memory ACTIVE the second chunk differs from the
    /// same frames processed cold, and the difference is the head — the join is where the cold
    /// replicate pad used to be. (Zeros would be invariant, so this needs real content.)
    func testStreamingSecondChunkDiffersFromColdChunk() {
        let vae = SeedVR2VAE()
        let clip = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, 9, 32, 32], key: MLXRandom.key(7))
        let (head, tail) = (clip[0..., 0..., 0 ..< 5], clip[0..., 0..., 5 ..< 9])

        vae.resetStreamingMemory()
        _ = { let z = vae.encode(head, memoryState: .initializing)
              eval([vae.decode(z, memoryState: .initializing)] + vae.streamingMemoryTails()) }()
        let streamed = vae.decode(vae.encode(tail, memoryState: .active), memoryState: .active)

        vae.resetStreamingMemory()
        let cold = vae.decode(vae.encode(tail, memoryState: .initializing), memoryState: .initializing)
        eval(streamed, cold)

        XCTAssertEqual(streamed.shape[2], 4)
        XCTAssertNotEqual(cold.shape[2], streamed.shape[2],
                          "a cold 4-frame chunk decodes 4*(latT-1)+1 = 1 frame; streamed keeps 4")
        // And the memory bank is populated after a streaming pass but empty after a reset.
        XCTAssertFalse(vae.streamingMemoryTails().isEmpty)
        vae.resetStreamingMemory()
        XCTAssertTrue(vae.streamingMemoryTails().isEmpty)
    }

    /// **The reset-at-cut property (GAP-PROGRAM N11).** A chunk run as `.initializing` after
    /// `resetStreamingMemory()` mid-clip must be *bit-identical* to the same chunk run from a
    /// freshly constructed VAE — that equivalence is the whole basis for treating a scene cut as a
    /// clip boundary. If a reset left any residue, the post-cut shot would still be conditioned on
    /// the pre-cut one, which is the artefact the row exists to remove.
    func testResetMidClipIsEquivalentToColdStart() {
        let vae = SeedVR2VAE()
        let shotA = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, 9, 32, 32], key: MLXRandom.key(21))
        let shotB = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, 9, 32, 32], key: MLXRandom.key(22))

        // Run shot A so the memory bank is fully populated, then reset and run shot B.
        vae.resetStreamingMemory()
        let a = vae.decode(vae.encode(shotA, memoryState: .initializing), memoryState: .initializing)
        eval([a] + vae.streamingMemoryTails())
        XCTAssertFalse(vae.streamingMemoryTails().isEmpty, "shot A must leave state to clear")

        vae.resetStreamingMemory()
        XCTAssertTrue(vae.streamingMemoryTails().isEmpty, "the flush clears every conv's tail")
        let afterReset = vae.decode(vae.encode(shotB, memoryState: .initializing), memoryState: .initializing)

        // The same shot B through a VAE that never saw shot A at all.
        let fresh = SeedVR2VAE()
        fresh.resetStreamingMemory()
        let cold = fresh.decode(fresh.encode(shotB, memoryState: .initializing), memoryState: .initializing)
        eval(afterReset, cold)

        XCTAssertEqual(afterReset.shape, cold.shape)
        XCTAssertEqual(abs(afterReset - cold).max().item(Float.self), 0,
                       "a post-reset chunk must carry NOTHING from the previous shot")
    }

    /// T = 1 with the memory machinery present is bit-identical to T = 1 without it: `.disabled`
    /// takes the original replicate-pad expression and records nothing.
    func testSingleFrameUnaffectedByStreamingSurface() {
        let vae = SeedVR2VAE()
        let x = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, 1, 32, 32], key: MLXRandom.key(11))
        let a = vae.decode(vae.encode(x))
        let b = vae.decode(vae.encode(x, memoryState: .disabled), memoryState: .disabled)
        eval(a, b)
        XCTAssertEqual(abs(a - b).max().item(Float.self), 0)
        XCTAssertTrue(vae.streamingMemoryTails().isEmpty, "`.disabled` must record no state")
    }

    /// The latent-creator surfaces are T-general: noise sized to latentFrames, condition mask
    /// spanning all latent frames (SeedVR get_condition task="sr" semantics).
    func testLatentCreatorTemporalShapes() {
        let n1 = SeedVR2LatentCreator.noiseLatents(seed: 1, height: 8, width: 8)
        XCTAssertEqual(n1.shape, [1, 16, 1, 8, 8])
        let n4 = SeedVR2LatentCreator.noiseLatents(seed: 1, height: 8, width: 8, latentFrames: 4)
        XCTAssertEqual(n4.shape, [1, 16, 4, 8, 8])
        // Same seed, latentFrames 1: the default path draws the identical field (RNG stream pinned).
        XCTAssertEqual(abs(n1 - SeedVR2LatentCreator.noiseLatents(seed: 1, height: 8, width: 8, latentFrames: 1))
            .max().item(Float.self), 0)

        let cond = SeedVR2LatentCreator.condition(MLXArray.zeros([1, 16, 3, 8, 8], type: Float.self))
        XCTAssertEqual(cond.shape, [1, 17, 3, 8, 8])
        // Mask channel is ones over EVERY latent frame.
        XCTAssertEqual(cond[0..., 16 ..< 17].min().item(Float.self), 1)
    }
}
