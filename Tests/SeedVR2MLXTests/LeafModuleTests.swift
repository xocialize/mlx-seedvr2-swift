// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
import MLX
import XCTest

@testable import SeedVR2MLX

/// Shape/finite smoke tests for the mechanical leaf modules (run on CPU via xcodebuild).
/// Numerical parity vs goldens lands once the full transformer is assembled.
final class LeafModuleTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: Device(.cpu))
    }

    func testSwiGLUHiddenDimAndShape() {
        let mlp = SwiGLUMLP(dim: 2560, expandRatio: 4)  // hidden should be 6912 (matches checkpoint)
        let x = MLXRandom.normal([4, 2560])
        let y = mlp(x)
        eval(y)
        XCTAssertEqual(y.shape, [4, 2560])
        XCTAssertTrue(y.sum().item(Float.self).isFinite)
    }

    func testMMSwiGLUDualStream() {
        let mlp = MMSwiGLU(vidDim: 64, txtDim: 64, expandRatio: 4, isLastLayer: false)
        let (v, t) = mlp(MLXRandom.normal([2, 10, 64]), MLXRandom.normal([2, 5, 64]))
        eval(v, t)
        XCTAssertEqual(v.shape, [2, 10, 64])
        XCTAssertEqual(t.shape, [2, 5, 64])
    }

    func testTimeEmbeddingShape() {
        let te = TimeEmbedding(sinusoidalDim: 256, hiddenDim: 2560, outputDim: 15360)
        let out = te(MLXArray(Float(0.5)))
        eval(out)
        XCTAssertEqual(out.shape, [1, 15360])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    func testPatchInOutRoundTripShape() {
        // [B,C,T,H,W] -> tokens -> back; check the round-trip spatial dims.
        let pin = PatchIn(inChannels: 33, patchSize: [1, 2, 2], dim: 2560)
        let x = MLXRandom.normal([1, 33, 1, 8, 8])
        let (tokens, shape) = pin(x)
        eval(tokens)
        XCTAssertEqual(tokens.shape, [1, 16, 2560])  // 1*4*4 patches
        XCTAssertEqual(shape, [1, 4, 4])

        let pout = PatchOut(outChannels: 16, patchSize: [1, 2, 2], dim: 2560)
        let back = pout(tokens, vidShape: shape)
        eval(back)
        XCTAssertEqual(back.shape, [1, 16, 1, 8, 8])
    }

    func testAdaModulationShapes() {
        let ada = AdaModulation(dim: 64, isLastLayer: false)
        let hidden = MLXRandom.normal([2, 10, 64])
        let emb = MLXRandom.normal([2, 64, 2, 3])
        let mIn = ada.modulateVid(hidden, emb, .attn, .modIn)
        let mOut = ada.modulateVid(mIn, emb, .attn, .modOut)
        eval(mOut)
        XCTAssertEqual(mOut.shape, [2, 10, 64])
        // last-layer txt modulation is identity
        let adaLast = AdaModulation(dim: 64, isLastLayer: true)
        let same = adaLast.modulateTxt(hidden, emb, .mlp, .modIn)
        XCTAssertEqual(same.shape, [2, 10, 64])
    }
}
