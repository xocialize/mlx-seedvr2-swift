// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// The VAE's RGB conv_in (CausalConv3d 3 → 128) is a conv3d mlx 0.32.2 serves with `pad_and_slice_conv_3D_gpu`, which
// frees its zero fill value while the GPU still reads it (AB-T-0214). CausalConv3d routes it through ExactConv; on the
// GPU it must still match the convGeneral it used to run, on the CPU. Weights-free (seeded).
import Foundation
import MLX
import MLXExactConv
import MLXNN
import XCTest

@testable import SeedVR2MLX

final class PadHelperTests: XCTestCase {
    func testConvInIsRoutedOffThePadHelper() {
        let conv = CausalConv3d(3, 128)
        MLXRandom.seed(3)
        conv.update(parameters: conv.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
        let x = MLXRandom.uniform(low: -1, high: 1, [1, 3, 1, 40, 36])  // [B, C, T, H, W], one image frame

        let gpu = Device.withDefaultDevice(Device(.gpu)) { conv(x) }
        eval(gpu)
        // The pre-0.10.2 forward: replicate the first frame twice in front, then a plain convGeneral.
        var reference = MLXArray(0)
        withMLXCPU {
            let padded = concatenated([repeated(x[0..., 0..., 0 ..< 1], count: 2, axis: 2), x], axis: 2)
                .transposed(0, 2, 3, 4, 1)
            XCTAssertTrue(
                ExactConv.takesPadHelper3D(input: padded, weight: conv.weight),
                "a plain conv3d at this shape would take mlx's pad helper")
            reference = (convGeneral(padded, conv.weight, strides: [1, 1, 1], padding: [0, 1, 1]) + conv.bias)
                .transposed(0, 4, 1, 2, 3)
            eval(reference)
        }
        XCTAssertEqual(gpu.shape, [1, 128, 1, 40, 36])
        XCTAssertLessThan(abs(gpu - reference).max().item(Float.self), 1e-4)
    }
}
