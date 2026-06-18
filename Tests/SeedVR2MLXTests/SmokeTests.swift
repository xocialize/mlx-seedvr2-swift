// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
import MLX
import XCTest

@testable import SeedVR2MLX

final class SmokeTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // CPU stream needs no Metal default.metallib (sidesteps the SPM-CLI error)
        // and is the correct device for parity (Apple-GPU fp32 is tf32-like).
        Device.setDefault(device: Device(.cpu))
    }

    func testConfigDefaults() {
        XCTAssertEqual(SeedVR2Config.r3B.vidDim, 2560)
        XCTAssertEqual(SeedVR2Config.r3B.numLayers, 32)
        XCTAssertEqual(SeedVR2Config.r7B.vidDim, 3072)
        XCTAssertEqual(SeedVR2Config.r7B.numLayers, 36)
        XCTAssertFalse(SeedVR2Config.r7B.ropeOnText)
    }

    func testRMSNormRunsFinite() {
        let norm = SeedVR2RMSNorm(64)
        let x = MLXRandom.normal([2, 8, 64])
        let y = norm(x)
        eval(y)
        XCTAssertEqual(y.shape, [2, 8, 64])
        XCTAssertTrue(y.sum().item(Float.self).isFinite)
    }
}
