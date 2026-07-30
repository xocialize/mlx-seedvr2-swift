import Foundation
import MLXToolKit

/// Init-time configuration for `SeedVR2UpscalePackage` (C9) — shared by both the **image** and
/// **video** upscale surfaces. Stable for the session.
///
/// SeedVR2 ships **quant-as-repo**: `mlx-community/SeedVR2-3B-mlx` (fp16) and `…-mlx-int8`
/// (int8, near-lossless ~50.3 dB / cos 0.9999). int4 is deliberately *not* offered — it degrades
/// to ~22.7 dB. So `quant` selects the repo (unless `repoOverride` is set), and the engine charges
/// the matching `QuantFootprint` via `QuantConfigured`.
public struct SeedVR2Configuration: PackageConfiguration, ModelStorable, QuantConfigured, BudgetAware {
    /// fp16 or int8 only. int8 is the validated default (near-lossless, ~4.7 GB vs ~7.5 GB).
    public var quant: Quant
    /// Optional explicit weights repo; when `nil`, derived from `quant` (the canonical repos).
    public var repoOverride: String?
    /// Diffusion seed — deterministic output per seed (MLX-Swift RNG parity).
    public var seed: UInt64
    /// LAB-wavelet color transfer of the refined detail toward the pre-upscaled base (mflux parity).
    public var colorCorrect: Bool

    /// Default integer scale when a request omits one (2 or 4).
    public var defaultScale: Int
    /// Refinement tile size / overlap, shared by the **video** and (since the V10 fix) the **image**
    /// surface: the diffusion runs per tile at 1:1 after the pre-upscale, feathered seams via the shared
    /// `MLXTileProcessor`.
    public var tileSize: Int
    public var tileOverlap: Int

    /// **Halo (context padding) for the image surface's tiled refine, in output pixels per side.**
    ///
    /// Each tile is refined over `(tileSize + 2·tileHalo)²` of REAL surrounding image content and
    /// the margin is cropped off the output — the windowed attention sees the tile's neighbourhood
    /// instead of a border. Overlap blends outputs that already disagree; halo makes the calls see
    /// the same neighbourhood so they *could* agree.
    ///
    /// 🚨 **MEASURED NULL — do not re-try as a seam fix** (2026-07-30, GAP-PROGRAM V10-fix,
    /// `probes/v10fix_halo.out`). Swept 0/32/64/96 px on the flattest graphic + a photo master:
    /// every artefact axis flat (seam comb, 224-stride lattice power, chroma HF, flat-MAE) at up
    /// to **2.5× the run time and +1.2 GB**. The disproof, not just the null: the error folded on
    /// the 224 write grid is identical to ±0.03 across the sweep, and per-write-cell DC errors
    /// correlate **0.97–0.996 between halo 0 and halo 96** — the same tiles wrong the same way
    /// with 256² vs 448² of context. The residual is a deterministic, content-determined
    /// low-frequency bias with *global* in-window sensitivity; no finite halo can make two
    /// differently-placed calls agree. Kept because the machinery is correct and cheap at 0, and
    /// because the lever being present WITH its receipt is what stops the next re-derivation.
    ///
    /// ⚠️ Not free: at `tileSize` 256 a 64 px halo refines 384² per tile — 2.25× the pixels.
    /// Must keep `tileSize + 2·tileHalo` a multiple of 16 (VAE 8× + patch/window), i.e. `tileHalo`
    /// a multiple of 8. `0` (the default) is the pre-halo behaviour, byte-identical.
    public var tileHalo: Int

    /// 🚨 **Output-pixel ceiling for the image surface's single-pass path; above it, tile.**
    ///
    /// Measured 2026-07-29 (`mlxengine-todo/GAP-PROGRAM.md` **V10**). The stills path used to be
    /// single-pass at every size, and that **broke above ~1 MP output** — not gradually, as a cliff.
    /// SSIMULACRA2 against a lossless reference crop, one signage master:
    ///
    ///     128→512  ×4  →  −13.59     (works, looks good)
    ///     256→1024 ×4  →   −4.74     (works, looks good)
    ///     512→1024 ×2  →   +8.16     (works, looks good)
    ///     512→2048 ×4  →  −64.23     ← global softness, misregistration, glyphs destroyed
    ///
    /// 🔑 **Why a cliff and not a slope:** SeedVR2 runs windowed attention (`window: [4,3,3]`) over a
    /// latent 1/8 the pixel side, so a 2048² single pass is a **256²-token latent** — far outside its
    /// training regime. Tiling at `tileSize` keeps every diffusion call in-regime. It also bounds the
    /// activation peak, which was **43.7 GB attributed process footprint** for one 512→2048 still.
    ///
    /// Default `1024 × 1024`: the largest output measured clean single-pass. ⚠️ That is **one image's**
    /// cliff edge — it is a conservative default, not a characterized limit. `0` forces tiling always.
    public var imageWholeFramePixels: Int

    /// Absolute path to a pre-materialized weights snapshot (the directory holding
    /// `transformer.safetensors` / `vae.safetensors` / `pos_emb.safetensors` / `config.json`).
    /// **Honored OVER the engine-stamped `modelsRootDirectory`** — a stamped root is *appended to*
    /// for the HF download, which would corrupt an already-absolute path (the Anima v0.1.1 lesson).
    public var snapshotDirectory: URL?

    /// Where weights are materialized — set by the engine from its `ModelStore.root` (`ModelStorable`).
    /// `nil` → the core's default cache (`~/Library/Caches/seedvr2-mlx`).
    public var modelsRootDirectory: URL?

    /// Real headroom this model is loading into, stamped by the governor at load time (`BudgetAware`).
    /// `nil` → no figure; load the configured `quant`.
    public var availableBudgetBytes: UInt64?

    public init(quant: Quant = .int8,
                repoOverride: String? = nil,
                seed: UInt64 = 0,
                colorCorrect: Bool = true,
                defaultScale: Int = 2,
                tileSize: Int = 256,
                tileOverlap: Int = 32,
                tileHalo: Int = 0,
                imageWholeFramePixels: Int = 1024 * 1024,
                snapshotDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil,
                availableBudgetBytes: UInt64? = nil) {
        self.quant = quant
        self.repoOverride = repoOverride
        self.seed = seed
        self.colorCorrect = colorCorrect
        self.defaultScale = defaultScale
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.tileHalo = tileHalo
        self.imageWholeFramePixels = imageWholeFramePixels
        self.snapshotDirectory = snapshotDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.availableBudgetBytes = availableBudgetBytes
    }

    /// The canonical repo for a quant (fp16 / int8 only).
    public static func repo(for quant: Quant) -> String {
        quant == .fp16 ? "mlx-community/SeedVR2-3B-mlx" : "mlx-community/SeedVR2-3B-mlx-int8"
    }

    /// The effective weights repo for this configuration (override wins, else quant-derived).
    public var repo: String { repoOverride ?? Self.repo(for: quant) }

    // Persist only the portable knobs; environment-specific fields (stamped roots, budget, an
    // absolute snapshot path) are excluded from `Codable` — the engine re-stamps them per session.
    private enum CodingKeys: String, CodingKey {
        case quant, repoOverride, seed, colorCorrect, defaultScale, tileSize, tileOverlap
        case tileHalo, imageWholeFramePixels
    }
}
