import ArgumentParser
import Foundation
import MLX
import MLXToolKit
import MLXSeedVR2

/// Drive the conformant `SeedVR2UpscalePackage` exactly as the engine would: license gate → init →
/// load() → run(ImageUpscaleRequest) → write the upscaled PNG. Proves the package envelope and
/// reports the MLX activation peak for the footprint declaration (watchdog-safe component gate;
/// the in-app phys_footprint is the admission basis and reads ~2.5–2.9× higher — re-baseline there).
@main
struct PackageSmoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "seedvr2-package-smoke",
        abstract: "Drive SeedVR2UpscalePackage through load()/run() on one image.")

    @Option(name: .long, help: "Local snapshot dir (transformer/vae/pos_emb/config). Overrides repo download.")
    var snapshot: String?
    @Option(name: .long, help: "Input image path (png/jpeg).")
    var image: String
    @Option(name: .long, help: "Output PNG path.")
    var out: String
    @Option(name: .long, help: "Scale factor: 2 or 4.")
    var scale: Int = 2
    @Option(name: .long, help: "Quant: int8 (default) or fp16.")
    var quant: String = "int8"
    @Flag(name: .long, help: "Disable LAB color correction.")
    var noColorCorrect = false

    func run() async throws {
        let decl = SeedVR2UpscalePackage.manifest.license
        let gate = LicensePolicy.permissiveOnly.evaluate(decl)
        print("[pkg] license weight=\(decl.weightLicense) port=\(decl.portCodeLicense) → \(gate)")
        guard gate.isAdmitted else { throw ExitCode(1) }

        let q: Quant = quant == "fp16" ? .fp16 : .int8
        let cfg = SeedVR2Configuration(
            quant: q,
            colorCorrect: !noColorCorrect,
            snapshotDirectory: snapshot.map { URL(fileURLWithPath: $0) })

        let pkg = SeedVR2UpscalePackage(configuration: cfg)
        let loadStart = Date()
        try await pkg.load()
        MLX.GPU.clearCache()
        let resident = Double(MLX.GPU.activeMemory) / 1e9
        print(String(format: "[pkg] load → %.1fs, resident floor %.2f GB (quant=%@)",
                     Date().timeIntervalSince(loadStart), resident, quant))

        let data = try Data(contentsOf: URL(fileURLWithPath: image))
        let fmt: Image.Format = image.lowercased().hasSuffix(".png") ? .png : .jpeg
        let req = ImageUpscaleRequest(image: Image(format: fmt, data: data), scale: scale)

        MLX.GPU.resetPeakMemory()
        let runStart = Date()
        let resp = try await pkg.run(req)
        guard let r = resp as? ImageUpscaleResponse else { throw ExitCode(1) }
        try r.image.data.write(to: URL(fileURLWithPath: out))
        print(String(format: "[pkg] run → %dx%d scale=%d  (%.2fs, peak %.2f GB) → %@",
                     r.image.width ?? 0, r.image.height ?? 0, r.appliedScale,
                     Date().timeIntervalSince(runStart), Double(MLX.GPU.peakMemory) / 1e9, out))
    }
}
