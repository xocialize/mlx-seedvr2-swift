// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae/common/conv3d.py + SeedVR video_vae_v3 streaming memory.
import MLX
import MLXNN

/// VAE activation precision after group-norms (mflux `ModelConfig.precision`).
enum VAEPrecision { static let dtype: DType = .bfloat16 }

/// Streaming state for the causal VAE — SeedVR `video_vae_v3/modules/types.py` `MemoryState`.
///
/// A causal conv replicate-pads the FRONT of its sequence, so every chunk processed on its own
/// starts from a cold pad and is blind to the frames before it. `active` replaces that pad with
/// the tail of the previous chunk, which is what makes chunk N+1 continue chunk N instead of
/// restarting. `disabled` is the shipping single-chunk path and is bit-identical to the
/// pre-streaming code.
public enum VAEMemoryState: Sendable {
    /// No memory bank. The single-pass path — cold replicate pad, nothing recorded.
    case disabled
    /// First chunk of a clip: cold replicate pad (there is no previous chunk), tail recorded.
    case initializing
    /// A later chunk: the previous chunk's recorded tail replaces the replicate pad.
    case active
}

/// Reference box for a conv's streaming tail.
///
/// Deliberately NOT an `MLXArray` ivar: MLX-Swift's `Module.items()` mirrors ivars and maps any
/// `MLXArray` to `.parameters`, which would put the streaming tail into `parameters()`,
/// `update(parameters:)` and `eval(module)`. An opaque class lands in `.other` and stays out of
/// the parameter tree.
final class CausalMemory {
    var tail: MLXArray?
}

/// Causal (in time) 3D conv. Weight layout [O, kt, kh, kw, I] (MLX NDHWC) — load directly.
public final class CausalConv3d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let k: (Int, Int, Int), s: (Int, Int, Int), p: (Int, Int, Int)
    let causalTemporal: Bool, usePaddingCausal: Bool

    /// Frames of the (already head-extended) input this conv must carry into the next chunk —
    /// `-(stride_t - kernel_t)` from SeedVR `InflatedCausalConv3d.basic_forward`'s `mem_size`.
    /// k=3,s=1 -> 2 (exactly the replicate pad it replaces); k=3,s=2 (temporal down) -> 1;
    /// k=1 -> 0, no state at all.
    let memoryFrames: Int
    let memory = CausalMemory()

    public init(_ inCh: Int, _ outCh: Int, kernel: (Int, Int, Int) = (3, 3, 3),
                stride: (Int, Int, Int) = (1, 1, 1), padding: (Int, Int, Int) = (1, 1, 1),
                causalTemporal: Bool = true, usePaddingCausal: Bool = false) {
        self.k = kernel; self.s = stride; self.p = padding
        self.causalTemporal = causalTemporal; self.usePaddingCausal = usePaddingCausal
        self.memoryFrames = (causalTemporal && kernel.0 > 1) ? max(0, kernel.0 - stride.0) : 0
        self._weight.wrappedValue = MLXArray.zeros([outCh, kernel.0, kernel.1, kernel.2, inCh])
        self._bias.wrappedValue = MLXArray.zeros([outCh])
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray, memoryState: VAEMemoryState = .disabled) -> MLXArray {
        var x = xIn  // [B,C,T,H,W]
        var tPad = p.0
        if causalTemporal && k.0 > 1 {
            // SeedVR `basic_forward`: the memory, when present and ACTIVE, IS the head extension —
            // `extend_head(input, memory=self.memory)` concatenates it in place of the replicate
            // pad. Anything else (disabled / first chunk / no tail yet) takes the cold pad, which
            // is the original expression untouched.
            if memoryState == .active, let tail = memory.tail {
                x = concatenated([tail.asType(x.dtype), x], axis: 2)
            } else {
                let causalPad = usePaddingCausal ? 2 * p.0 : k.0 - 1
                if causalPad > 0 {
                    let first = x[0..., 0..., 0 ..< 1]
                    let pad = repeated(first, count: causalPad, axis: 2)
                    x = concatenated([pad, x], axis: 2)
                }
            }
            tPad = 0
        }
        // Record this chunk's tail for the next one (`input[:, :, mem_size:]` — taken AFTER the
        // head extension, so a chunk that itself ran on memory still hands on a full-width tail).
        if memoryState == .disabled {
            memory.tail = nil
        } else if memoryFrames > 0 {
            // `.contiguous()` is load-bearing, not tidiness: the tail is a strided slice of
            // [B,C,T,H,W], and re-reading it strided in the next chunk's `concatenated` costs
            // real time. Measured at 256²/T=5: 0.391 -> 0.257 s/frame (-34 %). It buys little
            // memory on its own (21.61 -> 21.24 GiB uncapped) — the memory lever is WHEN the
            // tails are evaluated, see ``SeedVR2VAE/settle(_:_:_:)``. (SeedVR's own slicing path
            // does the same, `.detach().contiguous()`.)
            memory.tail = x[0..., 0..., (x.shape[2] - memoryFrames)...].contiguous()
        }
        x = x.transposed(0, 2, 3, 4, 1).asType(weight.dtype)        // NDHWC
        var out = convGeneral(x, weight, strides: [s.0, s.1, s.2], padding: [tPad, p.1, p.2])
        out = out + bias
        return out.transposed(0, 4, 1, 2, 3)                        // back to [B,O,T,H,W]
    }
}

/// GroupNorm over channels (input [B,C,T,H,W]) done in fp32, cast back to VAE precision.
///
/// 🚨 The statistic is PER FRAME, not per chunk. SeedVR's `causal_norm_wrapper`
/// (`causal_inflation_lib.py`) reshapes a 5-D input `b c t h w -> (b t) c h w` before the
/// GroupNorm, so each frame is normalised by its own H·W statistics. Handing MLX-Swift's
/// `GroupNorm` a 5-D `[B,T,H,W,C]` instead reduces over every dim between batch and channel —
/// T·H·W — which makes each frame's normalisation depend on the whole chunk. At T == 1 the two
/// are the same reduction (T·H·W == H·W), which is why the mflux image port never saw it. (The
/// VAE's own `Attention3D` already folds T into the batch dim before its group-norm — the two
/// paths in the same file disagreed.)
///
/// Measured at 256²/T=5 (V12-S): fixing it drops within-window max|ΔL*| 0.2744 → 0.1599, i.e.
/// below the content floor's own 0.2125, for no measurable cost (interleaved A/B, twice, capped
/// and uncapped: 0.3415/0.3419 before vs 0.3398/0.3498 s/frame after — B·T small norms instead of
/// one large one is not the win-or-lose one might expect).
/// ⚠️ It does NOT touch the chunk seam — boundary mean|ΔL*| 0.3395 → 0.3456, flat. The
/// "chunk-global statistic must show up at the join" reading was plausible and wrong; the seam
/// was the cold causal pad, which is what streaming memory fixes.
func vaeGroupNorm(_ x: MLXArray, _ norm: GroupNorm) -> MLXArray {
    let s = x.shape
    if s[2] == 1 {
        // T == 1: the original expression, verbatim — the shipping path stays bit-identical.
        var h = x.transposed(0, 2, 3, 4, 1)                          // NDHWC
        h = norm(h.asType(.float32)).asType(VAEPrecision.dtype)
        return h.transposed(0, 4, 1, 2, 3)
    }
    let (B, T) = (s[0], s[2])
    var h = x.transposed(0, 2, 3, 4, 1)                              // [B,T,H,W,C]
    h = h.reshaped([B * T, s[3], s[4], s[1]])                        // (b t) h w c — per-frame stats
    h = norm(h.asType(.float32)).asType(VAEPrecision.dtype)
    h = h.reshaped([B, T, s[3], s[4], s[1]])
    return h.transposed(0, 4, 1, 2, 3)
}
