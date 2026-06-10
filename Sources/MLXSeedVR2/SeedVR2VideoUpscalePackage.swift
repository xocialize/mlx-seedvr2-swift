import Foundation
import FormatBridge
import MLXToolKit
import SeedVR2MLX

/// Errors at the SeedVR2 package boundary.
public enum SeedVR2PackageError: Error {
    case unsupportedScale(Int)
}

/// An MLXEngine `videoUpscale` package over **SeedVR2-3B** (ByteDance, one-step diffusion SR) —
/// the first `Video → Video` transform of the visual optimization tier.
///
/// Per frame: CoreImage Lanczos pre-upscale (the spatial 2×/4×) → SeedVR2 one-step diffusion
/// **refinement** at 1:1, tile-blended with feathered seams → LAB-wavelet color transfer toward
/// the upscaled base (mflux parity). Frames stream decode→refine→encode (HEVC, BT.709-tagged)
/// so memory stays bounded; cancellation is honored per frame (C13 — the MemoryGovernor can
/// preempt between frames).
///
/// A thin conformance wrapper: the diffusion pipeline lives in `seedvr2-mlx-swift` (e2e GPU/int8
/// validated via Forge); the tiling machinery is shared from `realesrgan-mlx-swift`.
@InferenceActor
public final class SeedVR2VideoUpscalePackage: ModelPackage {
    public typealias Configuration = SeedVR2Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // SeedVR2 weights: Apache-2.0 (ByteDance-Seed). Port code: MIT.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/SeedVR2-3B-mlx-int8",
                                   revision: "main", tier: 2),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .int8, residentBytes: 6_000_000_000),   // int8 (default)
                    QuantFootprint(quant: .bf16, residentBytes: 10_000_000_000),  // bf16 repo
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // 3B-param diffusion per tile per frame — heavy lift.
                chipFloor: .pro
            ),
            specialties: [],
            surfaces: [
                VideoUpscaleContract.descriptor(
                    name: "seedvr2-upscale",
                    summary: "SeedVR2-3B one-step diffusion video super-resolution (2x/4x, tile-refined, HEVC out)."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var refiner: SeedVR2FrameRefiner?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard refiner == nil else { return }
        // Materialize the snapshot (engine model store when set, else the core's default cache),
        // then load weights (int8 repos re-quantize the module tree before load).
        let upscaler: SeedVR2Upscaler
        if let root = configuration.modelsRootDirectory {
            let dir = try HFHub.snapshot(repoId: configuration.repo,
                                         cacheDir: root.appending(path: "SeedVR2", directoryHint: .isDirectory))
            upscaler = try SeedVR2Upscaler(directory: dir)
        } else {
            upscaler = try SeedVR2Upscaler(repoId: configuration.repo)
        }
        refiner = SeedVR2FrameRefiner(upscaler: upscaler,
                                      tileSize: configuration.tileSize,
                                      tileOverlap: configuration.tileOverlap,
                                      colorCorrect: configuration.colorCorrect,
                                      seed: configuration.seed)
    }

    public func unload() async {
        refiner = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let refiner else { throw PackageError.notLoaded }
        guard request.capability == .videoUpscale,
              let req = request as? VideoUpscaleRequest else {
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

        // Layer-2 media service (format-bridge): tier-agnostic decode — native codecs via
        // VideoToolbox, non-native (WebM/MKV/VP9/AV1…) in software — then HEVC/BT.709 encode.
        let meta = try await FrameStreamTransform.run(
            input: inURL, output: outURL, timing: .preserveSource
        ) { frame in
            try Task.checkCancellation()
            return [try refiner.refine(frame, factor: scale)]
        }

        let data = try Data(contentsOf: outURL)
        return VideoUpscaleResponse(
            video: Video(format: .mp4, data: data,
                         durationSeconds: meta.sourceDuration, frameRate: meta.sourceFrameRate),
            appliedScale: scale)
    }
}

extension SeedVR2VideoUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(SeedVR2VideoUpscalePackage.self)
    }
}
