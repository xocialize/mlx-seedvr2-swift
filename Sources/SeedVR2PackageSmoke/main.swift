import ArgumentParser
import Foundation
import MLX
import MLXEngineTestKit
import MLXServeCore
import MLXToolKit
import MLXSeedVR2

/// Drive the conformant `SeedVR2UpscalePackage` exactly as the engine would: license gate → init →
/// load() → run(ImageUpscaleRequest) → write the upscaled PNG. Proves the package envelope and
/// reports the MLX activation peak for the footprint declaration (watchdog-safe component gate;
/// the in-app phys_footprint is the admission basis and reads ~2.5–2.9× higher — re-baseline there).
///
/// `--engine-store <dir>` drives it through the real `MLXServeEngine` instead: register → prepare
/// against a model store at `<dir>` (the engine materializes the declared weight source into the
/// store's flat layout first, contract 1.24) → run. Prints `MLXEngineTestKit`'s `[MAT]` line and the
/// resulting store layout; `--expect-download yes|no` turns the download-phase observation into an
/// exit status (a fresh store must download; a store holding a v0.9.x snapshot must not).
@main
struct PackageSmoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "seedvr2-package-smoke",
        abstract: "Drive SeedVR2UpscalePackage through load()/run() on one image.")

    @Option(name: .long, help: "Local snapshot dir (transformer/vae/pos_emb/config). Overrides repo download.")
    var snapshot: String?
    @Option(name: .long, help: "Input image path (png/jpeg).")
    var image: String?
    @Option(name: .long, help: "Input video path (mp4/mov/m4v) — drives the videoUpscale surface instead.")
    var video: String?
    @Option(name: .long, help: "Output path (PNG for --image, MP4 for --video).")
    var out: String
    @Option(name: .long, help: "Scale factor: 2 or 4.")
    var scale: Int = 2
    @Option(name: .long, help: "Quant: int8 (default) or fp16.")
    var quant: String = "int8"
    @Flag(name: .long, help: "Disable LAB color correction.")
    var noColorCorrect = false
    @Option(name: .long, help: "Model store root: drive register → prepare → run through MLXServeEngine (the engine materializes the weights here).")
    var engineStore: String?
    @Option(name: .long, help: "With --engine-store: require that prepare did (yes) or did not (no) surface a .downloading phase.")
    var expectDownload: String?

    func run() async throws {
        // Line-buffer stdout: a multi-GB download into a redirected log must stay visible live.
        setvbuf(stdout, nil, _IOLBF, 0)

        let decl = SeedVR2UpscalePackage.manifest.license
        let gate = LicensePolicy.permissiveOnly.evaluate(decl)
        print("[pkg] license weight=\(decl.weightLicense) port=\(decl.portCodeLicense) → \(gate)")
        guard gate.isAdmitted else { throw ExitCode(1) }

        let q: Quant = quant == "fp16" ? .fp16 : .int8
        let cfg = SeedVR2Configuration(
            quant: q,
            colorCorrect: !noColorCorrect,
            snapshotDirectory: snapshot.map { URL(fileURLWithPath: $0) })
        let capability: Capability = video != nil ? .videoUpscale : .imageUpscale

        let execute: (any CapabilityRequest) async throws -> any CapabilityResponse
        if let engineStore {
            execute = try await prepareThroughEngine(cfg, capability: capability,
                                                     root: URL(fileURLWithPath: engineStore, isDirectory: true))
        } else {
            let pkg = SeedVR2UpscalePackage(configuration: cfg)
            let loadStart = Date()
            try await pkg.load()
            MLX.GPU.clearCache()
            let resident = Double(MLX.GPU.activeMemory) / 1e9
            print(String(format: "[pkg] load → %.1fs, resident floor %.2f GB (quant=%@)",
                         Date().timeIntervalSince(loadStart), resident, quant))
            execute = { try await pkg.run($0) }
        }

        if let video {
            // videoUpscale surface — used by the V10-fix colour-match temporal A/B
            // (SEEDVR2_VIDEO_GLOBAL_CC=1 flips the frame refiner to a per-frame global match).
            let data = try Data(contentsOf: URL(fileURLWithPath: video))
            let fmt: Video.Format = video.lowercased().hasSuffix(".mov") ? .mov : .mp4
            let req = VideoUpscaleRequest(video: Video(format: fmt, data: data), scale: scale)
            MLX.GPU.resetPeakMemory()
            let runStart = Date()
            let resp = try await execute(req)
            guard let r = resp as? VideoUpscaleResponse else { throw ExitCode(1) }
            try r.video.data.write(to: URL(fileURLWithPath: out))
            print(String(format: "[pkg] video run → scale=%d %.1fs dur=%.2fs fps=%.2f (peak %.2f GB) → %@",
                         r.appliedScale, Date().timeIntervalSince(runStart),
                         r.video.durationSeconds ?? 0, r.video.frameRate ?? 0,
                         Double(MLX.GPU.peakMemory) / 1e9, out))
            return
        }
        guard let image else {
            print("[pkg] pass --image or --video"); throw ExitCode(2)
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: image))
        let fmt: Image.Format = image.lowercased().hasSuffix(".png") ? .png : .jpeg
        let req = ImageUpscaleRequest(image: Image(format: fmt, data: data), scale: scale)

        MLX.GPU.resetPeakMemory()
        let runStart = Date()
        let resp = try await execute(req)
        guard let r = resp as? ImageUpscaleResponse else { throw ExitCode(1) }
        try r.image.data.write(to: URL(fileURLWithPath: out))
        print(String(format: "[pkg] run → %dx%d scale=%d  (%.2fs, peak %.2f GB) → %@",
                     r.image.width ?? 0, r.image.height ?? 0, r.appliedScale,
                     Date().timeIntervalSince(runStart), Double(MLX.GPU.peakMemory) / 1e9, out))
    }

    /// register → prepare through `MLXServeEngine` with a model store at `root`; returns the run seam.
    private func prepareThroughEngine(_ cfg: SeedVR2Configuration, capability: Capability,
                                      root: URL) async throws -> (any CapabilityRequest) async throws -> any CapabilityResponse {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let engine = MLXServeEngine()
        await engine.useModelStore(ModelStore(root: root))
        let id = try await engine.register(SeedVR2UpscalePackage.registration, configuration: cfg)

        // The configuration as the engine sees it at prepare (store root stamped).
        var stamped = cfg
        stamped.modelsRootDirectory = root
        let repo = stamped.effectiveRepo
        let flat = ModelStore(root: root).directory(for: repo)!
        let legacy = SeedVR2Configuration.legacyStoreDirectory(storeRoot: root, repo: repo)
        let needs = await engine.needsDownload(capability, package: id)
        print("[pkg] engine store=\(root.path) repo=\(repo) needsDownload=\(needs) "
              + "missing=\(stamped.missingWeightSources(storeRoot: root).map(\.role)) "
              + "found=\(stamped.existingWeightsDirectory(storeRoot: root)?.path ?? "none")")

        // Live download progress (the bench only tallies it): one line per 5%.
        let printer = Task { @MainActor in
            var lastBucket = -1
            while !Task.isCancelled {
                if case .downloading(let fraction, let bps) = engine.preparation.phase(for: capability,
                                                                                      package: id.description) {
                    let bucket = Int(fraction * 20)
                    if bucket != lastBucket {
                        lastBucket = bucket
                        print(String(format: "[pkg] .downloading %5.1f%%  %@", fraction * 100,
                                     bps.map { String(format: "%.1f MB/s", $0 / 1e6) } ?? "—"))
                    }
                }
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
        }
        let mat = try await MaterializationBench.run(
            engine: engine, capability: capability, package: id, configuration: stamped,
            sourceRepo: SeedVR2UpscalePackage.manifest.provenance.sourceRepo, storeRoot: root)
        printer.cancel()
        print(mat.logLine)

        let sizes = SeedVR2Configuration.weightFiles.map { file -> String in
            let size = (try? FileManager.default.attributesOfItem(
                atPath: flat.appending(path: file).path)[.size] as? NSNumber)?.int64Value ?? 0
            return "\(file)=\(size)"
        }
        let legacyPresent = FileManager.default.fileExists(atPath: legacy.path)
        print("[pkg] layout flat=\(flat.path) [\(sizes.joined(separator: " "))] "
              + "legacy=\(legacyPresent ? "PRESENT" : "absent") "
              + "missingAfter=\(stamped.missingWeightSources(storeRoot: root).map(\.role))")

        if let expectDownload {
            let want = expectDownload == "yes"
            guard mat.sawDownloadingPhase == want else {
                print("[pkg] ❌ expected downloadPhase=\(want ? "yes" : "NO"), observed \(mat.sawDownloadingPhase ? "yes" : "NO")")
                throw ExitCode(3)
            }
            print("[pkg] ✅ downloadPhase=\(want ? "yes" : "NO") as expected")
        }
        return { try await engine.run($0, package: id) }
    }
}
