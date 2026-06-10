import Foundation
import MLXToolKit

/// Init-time configuration for `SeedVR2VideoUpscalePackage` (C9).
public struct SeedVR2Configuration: PackageConfiguration, ModelStorable {
    /// HF repo holding the SeedVR2 MLX weights. int8 is the validated default (≈half the bf16
    /// footprint, parity-validated via the Forge e2e run); `mlx-community/SeedVR2-3B-mlx` is bf16.
    public var repo: String
    /// Default integer scale when a request doesn't specify one (2 or 4).
    public var defaultScale: Int
    /// Refinement tile size / overlap (the diffusion runs per tile at 1:1 after the pre-upscale).
    public var tileSize: Int
    public var tileOverlap: Int
    /// LAB-wavelet color transfer of the refined detail toward the pre-upscaled base (W4 parity).
    public var colorCorrect: Bool
    /// Diffusion seed (deterministic output per seed).
    public var seed: UInt64
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default cache. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/SeedVR2-3B-mlx-int8",
                defaultScale: Int = 2,
                tileSize: Int = 256,
                tileOverlap: Int = 32,
                colorCorrect: Bool = true,
                seed: UInt64 = 0,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.defaultScale = defaultScale
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.colorCorrect = colorCorrect
        self.seed = seed
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, defaultScale, tileSize, tileOverlap, colorCorrect, seed
    }
}
