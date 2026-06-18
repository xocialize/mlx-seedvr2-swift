// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Pipeline parity: seeded-noise RNG match, 1-step scheduler, decode wiring. CPU.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import SeedVR2MLX

final class PipelineParityTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    func goldens() throws -> (URL, [String: MLXArray]) {
        guard let dir = AttentionParityTests.dir else { throw XCTSkip("weights not found") }
        return (dir, try loadArrays(url: dir.appendingPathComponent("goldens_pipeline.safetensors")))
    }

    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// MLX-Swift and MLX-Python share the RNG core — seeded noise must match exactly.
    func testSeededNoiseMatchesPython() throws {
        let (_, g) = try goldens()
        let noise = SeedVR2LatentCreator.noiseLatents(seed: 42, height: 90, width: 120)
        eval(noise)
        XCTAssertEqual(noise.shape, g["noise_90x120"]!.shape)
        let e = maxAbs(noise, g["noise_90x120"]!)
        print("seeded-noise RNG match: max_abs=\(e)")
        XCTAssertLessThan(e, 1e-5, "RNG mismatch — noise would need to be injected")
    }

    /// 1-step euler reduces to latents - noise; verify against the captured step.
    func testSchedulerStep() throws {
        let (_, g) = try goldens()
        let sched = SeedVR2EulerScheduler(numInferenceSteps: 1)
        let out = sched.step(noise: g["scheduler_noise_pred"]!, timestepIdx: 0, latents: g["scheduler_latents_in"]!)
        eval(out)
        let e = maxAbs(out, g["latents_after_step"]!)
        print("scheduler step parity: max_abs=\(e)")
        XCTAssertLessThan(e, 1e-4, "scheduler diverges")
    }

    /// Full-resolution decode(latents_after_step) vs the NON-tiled mflux decode.
    /// NB: mflux's pipeline uses *tiled* VAE decode (`decoded_pre_color`), which differs
    /// from a single full decode by ~20% at tile boundaries — tiling is a host concern
    /// (ForgeUpscaler.MLXTileProcessor). This gates the wiring against the non-tiled oracle.
    func testDecodeWiring() throws {
        let (dir, g) = try goldens()
        let weights = try loadArrays(url: dir.appendingPathComponent("vae.safetensors"))
        let vae = SeedVR2VAE()
        try vae.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(vae)
        var decoded = vae.decode(g["latents_after_step"]!)   // [B,3,1,H,W]
        if decoded.ndim == 5 { decoded = decoded[0..., 0..., 0] }
        eval(decoded)
        let ref = g["decoded_nontiled"]!.ndim == 5 ? g["decoded_nontiled"]![0..., 0..., 0] : g["decoded_nontiled"]!
        let d = maxAbs(decoded.asType(.float32), ref.asType(.float32))
        let scale = abs(ref.asType(.float32)).max().item(Float.self)
        let rel = d / max(scale, 1e-6)
        print("decode wiring (vs non-tiled): rel_err=\(rel)")
        XCTAssertLessThan(rel, 5e-2, "decode wiring diverges")
    }
}
