// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Core SeedVR2 upscale: encode -> (noise + condition) -> 1-step transformer -> scheduler
// step -> decode. Preprocess (bicubic resize / softness) and LAB color-correction are
// host responsibilities (ForgeUpscaler / utilities) — this is the model critical path.
// Tiling for large images is delegated to the host (e.g. ForgeUpscaler.MLXTileProcessor).
import Foundation
import MLX
import MLXNN

public final class SeedVR2Upscaler {
    public let vae: SeedVR2VAE
    public let transformer: SeedVR2Transformer
    let textEmb: MLXArray   // [1,58,5120] precomputed positive embedding
    let config: SeedVR2Config

    /// Download + load from an HF repo id (e.g. `mlx-community/SeedVR2-3B-mlx-int8`).
    public convenience init(repoId: String, revision: String = "main") throws {
        try self.init(weights: SeedVR2Weights.from(repoId: repoId, revision: revision))
    }

    public convenience init(directory dir: URL) throws {
        try self.init(weights: SeedVR2Weights(directory: dir))
    }

    public init(weights w: SeedVR2Weights) throws {
        self.config = w.config
        self.transformer = SeedVR2Transformer(w.config)
        if let q = w.quantization {
            // Apply the same quantization the weights were saved with, then load.
            SeedVR2Quant.quantizeTransformer(transformer, groupSize: q.groupSize, bits: q.bits)
        }
        try transformer.update(parameters: ModuleParameters.unflattened(w.transformer), verify: .none)
        self.vae = SeedVR2VAE()
        try vae.update(parameters: ModuleParameters.unflattened(w.vae), verify: .none)
        self.textEmb = w.posEmb
        eval(transformer, vae)
    }

    /// processedImage: [B,3,H,W] (a single frame) or [B,3,T,H,W] (a causal frame stack,
    /// T ≡ 1 mod 4 for an exact latent count) in [-1,1], spatial dims padded to /16.
    /// Returns the decoded output, [B,3,T,H*?,W*?] — T frames in, T frames out.
    public func upscale(processedImage: MLXArray, seed: UInt64, numSteps: Int = 1) -> MLXArray {
        upscale(processedImage: processedImage, noise: nil, seed: seed, numSteps: numSteps)
    }

    /// Variant taking pre-made noise latents ([B,16,latT,h,w], matching the encoded latent's
    /// temporal and spatial dims — latT = 1 + (T-1)/4, h,w = H/8 × W/8 for the 8× VAE). A tiled
    /// host uses this to slice each tile's noise out of ONE field drawn over the whole image, so
    /// the field is continuous across tile seams instead of independently drawn (and therefore
    /// discontinuous) per tile. `noise: nil` draws from `seed` sized to this call's latent — the
    /// single-pass behaviour, unchanged.
    public func upscale(processedImage: MLXArray, noise: MLXArray?, seed: UInt64 = 0, numSteps: Int = 1) -> MLXArray {
        upscale(processedImage: processedImage, noise: noise, seed: seed, numSteps: numSteps,
                memoryState: .disabled)
    }

    /// Streaming variant: one chunk of a longer clip, with the causal VAE carrying its tail
    /// features across the join (SeedVR `video_vae_v3` `MemoryState`).
    ///
    /// Drive a clip as `.initializing` for the first chunk then `.active` for the rest, and call
    /// ``SeedVR2VAE/resetStreamingMemory()`` before starting another clip. Chunk lengths follow
    /// the causal arithmetic: the first chunk is T ≡ 1 mod 4 frames, every later chunk is a
    /// multiple of 4 (its latents no longer carry the first-frame special case). `.disabled` —
    /// what the two signatures above pass — is the single-pass path, unchanged.
    ///
    /// 🚨 **A scene cut is a clip boundary, and the driver must END THE CHUNK AT IT** — not merely
    /// reset at the next chunk join. Measured (GAP-PROGRAM N11 Task 3,
    /// `mlxengine-todo/probes/n11_reset_at_cut.out`), splicing two shots and comparing against the
    /// T=1 control, mean |ΔL*| over the 8 frames after the cut:
    ///
    /// | driver behaviour | post-cut |ΔL*| |
    /// |---|---|
    /// | no reset — carries shot A's tail into shot B | 4.13 |
    /// | reset, cut landed **mid-chunk** (flush at the next join) | 2.83 |
    /// | reset, chunk **ended at the cut** | **1.37** |
    ///
    /// The contamination is visible, not just numeric: the first frames of the new shot inherit the
    /// previous shot's colour cast and a ghosted luminance smear, decaying over ~10 frames. A
    /// mid-chunk cut cannot be repaired by flushing afterwards — the frames inside the straddling
    /// chunk were *jointly* processed with the old shot and come out bit-identical to the no-reset
    /// arm. So the cut detector drives **chunking**, not just the reset.
    ///
    /// ⚠️ The 4k+1 arithmetic does not close under splitting: if the whole clip is exactly
    /// expressible then the two segments either side of a cut cannot both be, so a cut always
    /// leaves one ragged segment. Pad its final chunk with a repeat of the last real frame and trim
    /// the extra outputs — never feed a padded frame forward into another chunk's memory.
    ///
    /// ⚠️ Only the VAE streams; the diffusion transformer still sees one chunk at a time. Pass
    /// `noise: nil` (or the same `seed`) per chunk — do NOT reach for the tiled stills path's
    /// one-field-sliced-per-tile trick here. Measured (V12-S §4c) it is WORSE on both statistics,
    /// because a fixed seed already gives every steady-state chunk the byte-identical field,
    /// which is maximally stable; one long field gives each chunk DIFFERENT noise.
    public func upscale(processedImage: MLXArray, noise: MLXArray?, seed: UInt64 = 0,
                        numSteps: Int = 1, memoryState: VAEMemoryState) -> MLXArray {
        let initial = vae.encode(processedImage, memoryState: memoryState)  // [B,16,latT,h,w]
        let condition = SeedVR2LatentCreator.condition(initial)  // [B,17,latT,h,w]
        if let noise {
            precondition(noise.shape[2] == initial.shape[2]
                         && noise.shape[3] == initial.shape[3] && noise.shape[4] == initial.shape[4],
                         "noise \(noise.shape) does not match the encoded latent \(initial.shape)")
        }
        var latents = noise ?? SeedVR2LatentCreator.noiseLatents(seed: seed, height: initial.shape[3], width: initial.shape[4],
                                                                 latentFrames: initial.shape[2])

        let scheduler = SeedVR2EulerScheduler(numInferenceSteps: numSteps)
        for t in 0 ..< scheduler.numSteps {
            let modelInput = concatenated([latents, condition], axis: 1)   // [B,33,latT,h,w]
            let pred = transformer(modelInput, textEmb, timestep: scheduler.timesteps[t])
            latents = scheduler.step(noise: pred, timestepIdx: t, latents: latents)
            eval(latents)
        }
        // `vae.encode` / `vae.decode` each settle their own streaming tails — see
        // ``SeedVR2VAE/settle(_:_:_:)`` for why the timing matters.
        return vae.decode(latents, memoryState: memoryState)
    }
}
