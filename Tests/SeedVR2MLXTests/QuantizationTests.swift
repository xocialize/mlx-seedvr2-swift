// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Produces + verifies the int8 transformer: quantize, gate cosine(t_out_int8, golden t_out),
// and write a self-contained `SeedVR2-3B-mlx-int8/` dir (transformer-int8 + vae fp16 + pos_emb
// + config) for publishing. Quant quality is gated on per-pass cosine (skill rule for
// generative models), not PSNR. CPU.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import SeedVR2MLX

final class QuantizationTests: XCTestCase {
    // Quant-QUALITY check (not oracle parity), so GPU is fine and ~10x faster than the CPU
    // stream. Compare int8 vs fp16 on the SAME device to isolate the quantization effect.

    func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.asType(.float32).flattened(), bf = b.asType(.float32).flattened()
        let dot = (af * bf).sum().item(Float.self)
        let na = sqrt((af * af).sum().item(Float.self)), nb = sqrt((bf * bf).sum().item(Float.self))
        return dot / (na * nb)
    }

    func testInt8QuantizeVerifyAndExport() throws {
        guard let dir = AttentionParityTests.dir else { throw XCTSkip("weights not found") }
        let (bits, gs) = (8, 64)
        let g = try loadArrays(url: dir.appendingPathComponent("goldens_transformer.safetensors"))

        // 1. fp16 reference forward (GPU)
        let fpWeights = try loadArrays(url: dir.appendingPathComponent("transformer.safetensors"))
        let fp = SeedVR2Transformer(.r3B)
        try fp.update(parameters: ModuleParameters.unflattened(fpWeights), verify: .none)
        let outFp = fp(g["t_vid_in"]!, g["t_txt_in"]!, timestep: g["t_timestep"]!)
        eval(outFp)

        // 2. quantize a fresh model int8 + forward
        let model = SeedVR2Transformer(.r3B)
        try model.update(parameters: ModuleParameters.unflattened(fpWeights), verify: .none)
        SeedVR2Quant.quantizeTransformer(model, groupSize: gs, bits: bits)
        eval(model)
        let out = model(g["t_vid_in"]!, g["t_txt_in"]!, timestep: g["t_timestep"]!)
        eval(out)

        // 3. gate quant quality (per-pass cosine, skill rule for generative models)
        let cos = cosine(out, outFp)
        print("int8 transformer t_out cosine vs fp16 (same device): \(cos)")
        XCTAssertGreaterThan(cos, 0.999, "int8 quant degrades t_out")

        // 3. export a self-contained int8 dir for publishing
        let out8 = dir.deletingLastPathComponent().appendingPathComponent("SeedVR2-3B-mlx-int8")
        try? FileManager.default.createDirectory(at: out8, withIntermediateDirectories: true)
        var qparams = [String: MLXArray]()
        for (k, v) in model.parameters().flattened() { qparams[k] = v }
        eval(qparams)
        try save(arrays: qparams, url: out8.appendingPathComponent("transformer.safetensors"))
        for f in ["vae.safetensors", "pos_emb.safetensors"] {
            let dst = out8.appendingPathComponent(f)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: dir.appendingPathComponent(f), to: dst)
        }
        let cfg = """
        {"model_type":"seedvr2","variant":"seedvr2-3b","transformer_overrides":{},
         "pos_emb_shape":[58,5120],"dtype":"int8",
         "quantization":{"bits":\(bits),"group_size":\(gs)},
         "upstream":"ByteDance-Seed/SeedVR (Apache-2.0)","mlx_reference":"filipstrand/mflux"}
        """
        try cfg.write(to: out8.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let sz = (try? FileManager.default.attributesOfItem(atPath: out8.appendingPathComponent("transformer.safetensors").path)[.size] as? Int) ?? 0
        print("wrote int8 transformer: \((sz ?? 0) / 1_000_000) MB -> \(out8.path)")

        // 4. round-trip: reload the published int8 dir via the real loader (config-driven
        //    quantize-then-update — the path Forge/end users use) and confirm it runs + matches.
        let w = try SeedVR2Weights(directory: out8)
        XCTAssertNotNil(w.quantization, "config.json should declare quantization")
        let reloaded = SeedVR2Transformer(w.config)
        if let q = w.quantization { SeedVR2Quant.quantizeTransformer(reloaded, groupSize: q.groupSize, bits: q.bits) }
        try reloaded.update(parameters: ModuleParameters.unflattened(w.transformer), verify: .none)
        eval(reloaded)
        let outReload = reloaded(g["t_vid_in"]!, g["t_txt_in"]!, timestep: g["t_timestep"]!)
        eval(outReload)
        let cosReload = cosine(outReload, out)   // should be identical to the in-memory int8 run
        print("int8 reload round-trip cosine vs in-memory int8: \(cosReload)")
        XCTAssertGreaterThan(cosReload, 0.99999, "int8 reload path diverges")
    }
}
