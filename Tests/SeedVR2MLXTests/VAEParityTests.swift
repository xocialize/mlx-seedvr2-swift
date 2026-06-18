// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// VAE encode/decode parity vs mflux goldens, on CPU.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import SeedVR2MLX

final class VAEParityTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    func loadVAE(_ dir: URL) throws -> (SeedVR2VAE, [String: MLXArray]) {
        let weights = try loadArrays(url: dir.appendingPathComponent("vae.safetensors"))
        let vae = SeedVR2VAE()
        try vae.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(vae)
        let g = try loadArrays(url: dir.appendingPathComponent("goldens_vae.safetensors"))
        return (vae, g)
    }

    func relErr(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        let scale = abs(b.asType(.float32)).max().item(Float.self)
        return d / max(scale, 1e-6)
    }

    func testVAEEncodeParity() throws {
        guard let dir = AttentionParityTests.dir else { throw XCTSkip("weights not found") }
        let (vae, g) = try loadVAE(dir)
        let out = vae.encode(g["vae_enc_in"]!)
        eval(out)
        XCTAssertEqual(out.shape, g["vae_enc_out"]!.shape)
        let e = relErr(out, g["vae_enc_out"]!)
        print("VAE encode parity: rel_err=\(e)")
        XCTAssertLessThan(e, 5e-2, "vae encode diverges")
    }

    func testVAEDecodeParity() throws {
        guard let dir = AttentionParityTests.dir else { throw XCTSkip("weights not found") }
        let (vae, g) = try loadVAE(dir)
        let out = vae.decode(g["vae_dec_in"]!)
        eval(out)
        XCTAssertEqual(out.shape, g["vae_dec_out"]!.shape)
        let e = relErr(out, g["vae_dec_out"]!)
        print("VAE decode parity: rel_err=\(e)")
        XCTAssertLessThan(e, 5e-2, "vae decode diverges")
    }
}
