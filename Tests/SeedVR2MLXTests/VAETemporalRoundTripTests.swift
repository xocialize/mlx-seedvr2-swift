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
