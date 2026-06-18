// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// The make-or-break gate: the full SeedVR2 transformer (all 32 blocks incl. shifted
// windows + shared-weight blocks + output ada) vs the mflux `t_out` golden, on CPU.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import SeedVR2MLX

final class TransformerParityTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    func testFullTransformerTOut() throws {
        guard let dir = AttentionParityTests.dir else { throw XCTSkip("weights not found; set SEEDVR2_WEIGHTS_DIR") }
        let weights = try loadArrays(url: dir.appendingPathComponent("transformer.safetensors"))
        let model = SeedVR2Transformer(.r3B)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(model)

        let g = try loadArrays(url: dir.appendingPathComponent("goldens_transformer.safetensors"))
        let out = model(g["t_vid_in"]!, g["t_txt_in"]!, timestep: g["t_timestep"]!)
        eval(out)

        let ref = g["t_out"]!
        XCTAssertEqual(out.shape, ref.shape)
        let err = abs(out.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        print("FULL TRANSFORMER t_out parity: max_abs=\(err)  shape=\(out.shape)")
        XCTAssertLessThan(err, 1e-1, "t_out diverges: \(err)")
    }
}
