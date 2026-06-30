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
    /// Refinement tile size / overlap for the **video** surface (the diffusion runs per tile at 1:1
    /// after the pre-upscale; feathered seams via the shared `MLXTileProcessor`). The image surface
    /// is single-pass V1.
    public var tileSize: Int
    public var tileOverlap: Int

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
    }
}
