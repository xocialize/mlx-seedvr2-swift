import Foundation
import FrameStreamNative
import MLX
import MLXToolKit
import SeedVR2MLX

/// Errors at the SeedVR2 package boundary.
public enum SeedVR2PackageError: Error {
    case unsupportedScale(Int)
}

/// An MLXEngine upscale package over **SeedVR2-3B** (ByteDance, one-step diffusion SR), exposing
/// **two** surfaces from one loaded model (the engine loads the 3B core once):
///
/// - **`imageUpscale`** — the diffusion / **Export-tier** backer of image upscale (Real-ESRGAN is
///   the fast-tier backer; multi-package per capability via `PackageID`). Single-pass V1.
/// - **`videoUpscale`** — the first `Video → Video` transform of the visual optimization tier:
///   frames stream decode → per-frame refine (tile-blended, feathered seams) → HEVC/BT.709 encode,
///   memory-bounded, cancellation honored per frame.
///
/// SeedVR2 is **1:1 spatially** (the model refines, it does not resize), so the scale factor is a
/// CoreImage Lanczos pre-upscale; the model then refines (+ optional LAB color transfer toward the
/// pre-upscaled base, mflux parity).
///
/// Born sweep-clean (1.14): split footprint per quant; `QuantConfigured` (fp16 + int8 only — int4
/// degrades); `BudgetAware` drops fp16→int8 under pressure (int8 is near-lossless); `unload()`
/// flushes the MLX pool. The diffusion pipeline lives in `seedvr2-mlx-swift` (`SeedVR2MLX`, e2e
/// GPU/int8 validated via Forge); the video tiling is shared from `realesrgan-mlx-swift`.
@InferenceActor
public final class SeedVR2UpscalePackage: ModelPackage {
    public typealias Configuration = SeedVR2Configuration

    /// fp16 needs ~7.5 GB resident + a multi-GB transient (≈11 GB working set); below this headroom
    /// `BudgetAware` substitutes the near-lossless int8 repo (measured ~4.72 GB resident).
    private static let fp16MinBudgetBytes: UInt64 = 11_000_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // SeedVR2 weights: Apache-2.0 (ByteDance-Seed). Port code: MIT. → permissive, no ack.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/SeedVR2-3B-mlx-int8",
                                   revision: "main", tier: 2),
            requirements: RequirementsManifest(
                footprints: [
                    // resident = measured weight floor (DiT + fp16 VAE, post-load clearCache); peak
                    // activation = one-step diffusion + VAE-decode transient. Measured via
                    // seedvr2-package-smoke @256²→2× (512²): int8 resident 4.72 GB, MLX-peak 7.07 GB
                    // (transient ~2.35 GB). int8 solid-measured; fp16 resident est from fp16/int8 DiT
                    // delta. Activation conservative (single-pass image / per-tile video are
                    // resolution-driven); in-app phys re-baseline pending (smoke MLX-peak under-reads
                    // admission phys ~2.5–2.9×).
                    QuantFootprint(quant: .int8, residentBytes: 4_800_000_000,
                                   peakActivationBytes: 3_500_000_000),   // int8 (default) — measured floor
                    QuantFootprint(quant: .fp16, residentBytes: 7_500_000_000,
                                   peakActivationBytes: 3_500_000_000),   // fp16 floor = est (not measured)
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // 3B-param one-step diffusion + temporal VAE — heavy lift.
                chipFloor: .pro
            ),
            specialties: [],
            // One loaded model, two surfaces (C1/C11).
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "seedvr2-upscale-image",
                    summary: "SeedVR2-3B one-step diffusion image super-resolution (2×/4×, Export tier)."),
                VideoUpscaleContract.descriptor(
                    name: "seedvr2-upscale-video",
                    summary: "SeedVR2-3B one-step diffusion video super-resolution (2×/4×, tile-refined, HEVC out)."),
            ]
        )
    }

    private let configuration: Configuration
    private var upscaler: SeedVR2Upscaler?
    private var imageRefiner: SeedVR2ImageRefiner?
    private var frameRefiner: SeedVR2FrameRefiner?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard upscaler == nil else { return }   // idempotent (C13: re-loadable after unload)

        // BudgetAware: int8 is near-lossless, so honor a tight stamped budget by substituting it.
        var quant = configuration.quant
        if quant == .fp16, let budget = configuration.availableBudgetBytes, budget < Self.fp16MinBudgetBytes {
            quant = .int8
        }

        let model: SeedVR2Upscaler
        if let snapshot = configuration.snapshotDirectory {
            // Absolute pre-materialized snapshot — honored OVER the stamped root (Anima lesson).
            model = try SeedVR2Upscaler(directory: snapshot)
        } else if let root = configuration.modelsRootDirectory {
            // Materialize into the engine model store via the core's own (swift-transformers-free)
            // downloader; per-repo subdir so fp16 and int8 snapshots don't collide.
            let repo = SeedVR2Configuration.repo(for: quant)
            let dir = try HFHub.snapshot(
                repoId: repo,
                cacheDir: root.appending(path: "seedvr2-mlx/\(repo.replacingOccurrences(of: "/", with: "--"))",
                                         directoryHint: .isDirectory))
            model = try SeedVR2Upscaler(directory: dir)
        } else {
            model = try SeedVR2Upscaler(repoId: SeedVR2Configuration.repo(for: quant))   // default cache
        }

        upscaler = model
        imageRefiner = SeedVR2ImageRefiner(upscaler: model,
                                           seed: configuration.seed,
                                           colorCorrect: configuration.colorCorrect)
        frameRefiner = SeedVR2FrameRefiner(upscaler: model,
                                           tileSize: configuration.tileSize,
                                           tileOverlap: configuration.tileOverlap,
                                           colorCorrect: configuration.colorCorrect,
                                           seed: configuration.seed)
    }

    public func unload() async {
        upscaler = nil
        imageRefiner = nil
        frameRefiner = nil
        // Dropping the refs alone leaves weight/activation buffers in MLX's pool, so phys_footprint
        // doesn't fall and engine.evict / R-MEM-1 can't reclaim — flush the pool.
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        switch request.capability {
        case .imageUpscale:  return try await runImage(request)
        case .videoUpscale:  return try await runVideo(request)
        default:             throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - imageUpscale (single-pass)

    private func runImage(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let imageRefiner else { throw PackageError.notLoaded }
        guard let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let scale = req.scale ?? configuration.defaultScale
        guard scale == 2 || scale == 4 else { throw SeedVR2PackageError.unsupportedScale(scale) }
        try Task.checkCancellation()

        let out = try imageRefiner.refine(req.image, factor: scale)
        return ImageUpscaleResponse(image: out, appliedScale: scale)
    }

    // MARK: - videoUpscale (streamed, tile-refined per frame)

    private func runVideo(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let frameRefiner else { throw PackageError.notLoaded }
        guard let req = request as? VideoUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let scale = req.scale ?? configuration.defaultScale
        guard scale == 2 || scale == 4 else { throw SeedVR2PackageError.unsupportedScale(scale) }
        try Task.checkCancellation()

        // Round-trip the container bytes through temp files (AVFoundation reads/writes URLs).
        let tmpDir = FileManager.default.temporaryDirectory
        let ext = req.video.format.rawValue
        let inURL = tmpDir.appendingPathComponent(UUID().uuidString).appendingPathExtension(ext)
        let outURL = tmpDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        try req.video.data.write(to: inURL)
        defer {
            try? FileManager.default.removeItem(at: inURL)
            try? FileManager.default.removeItem(at: outURL)
        }

        // FFmpeg-free streaming decode→transform→encode (frame-stream-native): AVFoundation
        // AVAssetReader (BGRA) → per-frame refine → HEVC/BT.709 encode, memory bounded. Native
        // containers only (mp4/mov/m4v); non-native input must be normalized upstream.
        let meta = try await NativeFrameStream.run(
            input: inURL, output: outURL, timing: .preserveSource
        ) { frame in
            try Task.checkCancellation()
            return [try frameRefiner.refine(frame, factor: scale)]
        }

        let data = try Data(contentsOf: outURL)
        return VideoUpscaleResponse(
            video: Video(format: .mp4, data: data,
                         durationSeconds: meta.sourceDuration, frameRate: meta.sourceFrameRate),
            appliedScale: scale)
    }
}

extension SeedVR2UpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(SeedVR2UpscalePackage.self)
    }
}
