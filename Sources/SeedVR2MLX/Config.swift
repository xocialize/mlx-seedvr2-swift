// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
import Foundation

/// SeedVR2 transformer configuration. Defaults match the 3B checkpoint
/// (mflux `SeedVR2Transformer.__init__`); 7B applies `transformerOverrides`.
///
/// 🚨 **`r7B` CANNOT LOAD A 7B CHECKPOINT. Verified 2026-07-29 against ByteDance's own
/// `seedvr2_ema_7b_sharp.pth`** — spike **V11** in `mlxengine-todo/GAP-PROGRAM.md` (probe + full output
/// under `mlxengine-todo/probes/`). The dimensions below are right; **two architectural facts this struct
/// cannot express make the load fail**, and neither is a naming difference:
///
///   1. **The 7B MLP is non-gated with biases.** `blocks.N.mlp.{txt,vid}.proj_{in,out}.bias` are the
///      *only* keys the 7B has that the 3B lacks, and `proj_in_gate` appears in all three 3B variants
///      and none of the 7B's. `SwiGLU.swift` is unconditionally gated with `bias: false`, and there is
///      no `mlpType` field to switch on.
///   2. **The 7B has no output adaLN and no final norm.** `vid_out_ada.out_scale`, `.out_shift` and
///      `vid_out_norm.weight` are present in the 3B and absent from the 7B, whose non-block keys are
///      only `emb_in.*`, `txt_in.*`, `vid_in.proj.*`, `vid_out.proj.*`. `Transformer.swift` declares all
///      three and *uses* them in its forward pass — so this is a forward-pass difference, not merely
///      missing parameters.
///
/// ✅ **`mmLayers = 36` below is CORRECT** — the 7B genuinely has zero `.all` blocks; all 36 are
/// multi-modal. And `txt_in` is 5120-dim in both checkpoints, so 3B text embeddings are reusable.
///
/// ⚠️ **A third claim in the first version of this note was WRONG and is recorded because the mistake
/// generalizes.** It read *"`emb_in.proj_in` takes a 256-dim time embedding (3B: 64)"*. **Both are 256.**
/// The "64" came from reading `[2560, 64]` in the **int8** conversion as a logical shape — MLX 8-bit packs
/// four values per `uint32`, so 64 × 4 = 256, and the `scales [2560, 4]` confirm it (group size 64).
/// 🔑 **Never infer an architecture from a quantized checkpoint's shapes:** the packing factor divides
/// them silently, and the result is always plausible. A control arm on a checkpoint whose key set is
/// already known is what caught it.
///
/// Until the branch is implemented, `WeightLoader.swift` will still select `r7B` from a `variant` string
/// containing `"7b"` and then fail to load. Fixing it is two local changes — a non-gated-MLP-with-biases
/// variant behind a config switch, and making the output adaLN + final norm conditional.
public struct SeedVR2Config: Codable, Sendable {
    public var vidInChannels: Int = 33
    public var vidOutChannels: Int = 16
    public var vidDim: Int = 2560
    public var txtInDim: Int = 5120
    public var heads: Int = 20
    public var headDim: Int = 128
    public var expandRatio: Int = 4
    public var ropeOnText: Bool = true
    public var normEps: Float = 1e-5
    public var patchSize: [Int] = [1, 2, 2]
    public var numLayers: Int = 32
    public var mmLayers: Int = 10
    public var ropeDim: Int = 128
    public var window: [Int] = [4, 3, 3]

    public static let r3B = SeedVR2Config()

    public static let r7B: SeedVR2Config = {
        var c = SeedVR2Config()
        c.vidDim = 3072; c.heads = 24; c.numLayers = 36
        c.mmLayers = 36; c.ropeDim = 64; c.ropeOnText = false
        return c
    }()

    /// Apply mflux's `transformer_overrides` dict (from exported config.json).
    public mutating func apply(overrides: [String: Int]) {
        if let v = overrides["vid_dim"] { vidDim = v }
        if let v = overrides["heads"] { heads = v }
        if let v = overrides["num_layers"] { numLayers = v }
        if let v = overrides["mm_layers"] { mmLayers = v }
        if let v = overrides["rope_dim"] { ropeDim = v }
        if let v = overrides["rope_on_text"] { ropeOnText = v != 0 }
    }
}
