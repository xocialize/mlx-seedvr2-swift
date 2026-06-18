// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// W4 parity: LAB-wavelet color transfer vs the mflux final_image. CPU.
import Foundation
import MLX
import XCTest

@testable import SeedVR2MLX

final class ColorCorrectParityTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    func testLabTransferVsFinalImage() throws {
        guard let dir = AttentionParityTests.dir else { throw XCTSkip("weights not found") }
        let g = try loadArrays(url: dir.appendingPathComponent("goldens_pipeline.safetensors"))

        // content = full (non-color-corrected) decode; style = preprocessed input; both [1,3,720,960] in [-1,1].
        let out = SeedVR2ColorCorrect.labTransfer(content: g["decoded_pre_color"]!,
                                                  style: g["processed_image"]!, luminanceWeight: 0.8)
        // -> uint8-scale [720,960,3] to compare with the golden final_image (float32 [0,255]).
        var img = clip((out + 1) * 0.5, min: 0, max: 1) * 255
        img = img[0].transposed(1, 2, 0)   // [720,960,3]
        eval(img)

        let ref = g["final_image"]!         // [720,960,3] in [0,255]
        XCTAssertEqual(img.shape, ref.shape)
        let diff = abs(img.asType(.float32) - ref.asType(.float32))
        let meanAbs = diff.mean().item(Float.self)
        let maxAbs = diff.max().item(Float.self)
        print("color-correct parity vs final_image: meanAbs=\(meanAbs) px, maxAbs=\(maxAbs) px")
        XCTAssertLessThan(meanAbs, 2.0, "color transfer diverges from mflux")
    }
}
