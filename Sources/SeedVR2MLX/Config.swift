// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
import Foundation

/// SeedVR2 transformer configuration. Defaults match the 3B checkpoint
/// (mflux `SeedVR2Transformer.__init__`); 7B applies `transformerOverrides`.
///
/// ⚠️ **`r7B` IS NOMINAL — no 7B-lineage checkpoint has ever been loaded through it, and as written it
/// very likely cannot be.** Recorded 2026-07-29 (`mlxengine-todo/PORT-QUEUE.md` P16, spike **V11**). The
/// dimensions here are right; three things this struct cannot express are not:
///
///   1. **The MLP is non-gated with biases** in the 7B family. `SwiGLU.swift` is unconditionally gated
///      with `bias: false`, and there is no `mlpType` field to switch on.
///   2. **`out_scale` / `out_shift` / `vid_out_norm` are absent** from 7B-lineage checkpoints, but
///      `Transformer.swift` declares all three and *uses* them in its forward pass.
///   3. **`emb_in.proj_in` takes a 256-dim time embedding** (3B: 64), which `apply(overrides:)` has no
///      field for.
///
/// Evidence is a safetensors key-pattern diff of `lvladikov/SeedVR2-1.4B` (a 6-block distill of
/// ByteDance's 7B) against the shipping `SeedVR2-3B-mlx-int8`. ⚠️ **Honest limit:** that infers the
/// teacher's architecture from a third party's distill of it — the distiller may have made these changes
/// themselves, in which case `r7B` is correct and the distill is the outlier. **V11 settles it against
/// ByteDance's own 7B.** Until then, treat `r7B` as an untested hypothesis rather than a supported path:
/// `WeightLoader.swift` will select it from a `variant` string containing `"7b"` and then fail to load.
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
