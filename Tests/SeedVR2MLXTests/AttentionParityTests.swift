// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Crux parity: MMAttention + TransformerBlock (block-0 config: non-shared, unshifted,
// non-last, rope_on_text) vs mflux goldens. Validates RoPE + Window + Attention together.
// Needs the exported weights/goldens from seedvr2-mlx/scripts (set SEEDVR2_WEIGHTS_DIR or
// place at ../seedvr2-mlx/dist/SeedVR2-3B-mlx). Run via xcodebuild (CPU).
import Foundation
import MLX
import MLXNN
import XCTest

@testable import SeedVR2MLX

final class AttentionParityTests: XCTestCase {
    static var dir: URL? {
        if let e = ProcessInfo.processInfo.environment["SEEDVR2_WEIGHTS_DIR"] { return URL(fileURLWithPath: e) }
        let guess = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("seedvr2-mlx/dist/SeedVR2-3B-mlx")
        return FileManager.default.fileExists(atPath: guess.appendingPathComponent("transformer.safetensors").path) ? guess : nil
    }

    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    func loadBlock0(_ dir: URL) throws -> (TransformerBlock, [String: MLXArray]) {
        let all = try loadArrays(url: dir.appendingPathComponent("transformer.safetensors"))
        var blk: [String: MLXArray] = [:]
        for (k, v) in all where k.hasPrefix("blocks.0.") { blk[String(k.dropFirst("blocks.0.".count))] = v }
        let block = TransformerBlock(isLastLayer: false)  // 3B defaults, block 0
        try block.update(parameters: ModuleParameters.unflattened(blk), verify: .none)
        eval(block)
        let g = try loadArrays(url: dir.appendingPathComponent("goldens_transformer.safetensors"))
        return (block, g)
    }

    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    func testAttentionBlock0Parity() throws {
        guard let dir = Self.dir else { throw XCTSkip("weights not found; set SEEDVR2_WEIGHTS_DIR") }
        let (block, g) = try loadBlock0(dir)
        let vs = g["vid_shape"]!.asArray(Int32.self).map { Int($0) }   // [1,45,60]
        let txtLen = Int(g["txt_shape"]!.asArray(Int32.self)[0])        // 58

        let (vidOut, txtOut) = block.attn(g["attn0_vid_in"]!, g["attn0_txt_in"]!,
                                          vidShape: [vs], txtLen: txtLen)
        eval(vidOut, txtOut)
        let ev = maxAbs(vidOut, g["attn0_vid_out"]!)
        let et = maxAbs(txtOut, g["attn0_txt_out"]!)
        print("attn block0 parity: vid max_abs=\(ev)  txt max_abs=\(et)")
        XCTAssertLessThan(ev, 2e-2, "attn vid diverges")
        XCTAssertLessThan(et, 2e-2, "attn txt diverges")
    }

    func testTransformerBlock0Parity() throws {
        guard let dir = Self.dir else { throw XCTSkip("weights not found; set SEEDVR2_WEIGHTS_DIR") }
        let (block, g) = try loadBlock0(dir)
        let vs = g["vid_shape"]!.asArray(Int32.self).map { Int($0) }
        let txtLen = Int(g["txt_shape"]!.asArray(Int32.self)[0])

        let (vidOut, txtOut) = block(g["blk0_vid_in"]!, g["blk0_txt_in"]!, emb: g["blk0_emb"]!,
                                     vidShape: [vs], txtLen: txtLen)
        eval(vidOut, txtOut)
        let ev = maxAbs(vidOut, g["blk0_vid_out"]!)
        let et = maxAbs(txtOut, g["blk0_txt_out"]!)
        print("block0 parity: vid max_abs=\(ev)  txt max_abs=\(et)")
        XCTAssertLessThan(ev, 3e-2, "block vid diverges")
        XCTAssertLessThan(et, 3e-2, "block txt diverges")
    }
}
