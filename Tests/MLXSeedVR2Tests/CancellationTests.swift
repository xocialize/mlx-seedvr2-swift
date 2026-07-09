// CancellationTests.swift — SeedVR2 through the engine's CAN gate (offline, no MLX kernels).
// ONE package, TWO surfaces (imageUpscale + videoUpscale): CAN-1/2 drive the real run()
// pre-cancelled on both surfaces (the entry checkpoint fires before capability dispatch or
// notLoaded validation); CAN-3 is the document of record for the checkpoint cadence:
//   • videoUpscale — per decoded SOURCE frame in the NativeFrameStream transform closure
//     (SeedVR2UpscalePackage.runVideo), plus per diffusion tile inside each frame's refine
//     (SeedVR2FrameRefiner.refine, MLXTileProcessor forward closure — throws propagate unchanged).
//   • imageUpscale — V1 is a single monolithic one-step diffusion eval (no tile loop): the real
//     seams are the entry checkpoint and the pre-forward checkpoint in SeedVR2ImageRefiner.refine.
// No do/catch is reachable from run(), so nothing can launder the CancellationError.

import Foundation
import MLXServeConformance
import MLXToolKit
import Testing
@testable import MLXSeedVR2

struct CancellationTests {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification (both surfaces)

    @Test func canGatePreCancelledImageRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // dispatch, validation, or weights are touched, so this is offline-safe.
        let package = SeedVR2UpscalePackage(configuration: SeedVR2Configuration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageUpscaleRequest(image: Image(format: .png, data: Data())))
        #expect(report.passed, "\(report.summary)")
    }

    @Test func canGatePreCancelledVideoRun() async {
        let package = SeedVR2UpscalePackage(configuration: SeedVR2Configuration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: VideoUpscaleRequest(video: Video(format: .mp4, data: Data())))
        #expect(report.passed, "\(report.summary)")
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    @Test func canCadenceDeclaration() {
        // videoUpscale is a long-run capability (and peak activation 4.5 GB ≥ 2 GB) —
        // the sub-second exemption is not available.
        #expect(CancellationConformance.longRunImplied(by: SeedVR2UpscalePackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: SeedVR2UpscalePackage.manifest,
            posture: .cadence([
                // videoUpscale: once per decoded source frame — the NativeFrameStream transform
                // closure in SeedVR2UpscalePackage.runVideo checks before each frame's refine.
                .init(phase: .upsample, unit: .frame),
                // Within each frame (and bounding the worst gap): once per 256² diffusion tile —
                // SeedVR2FrameRefiner.refine's MLXTileProcessor forward closure. The imageUpscale
                // surface has no tile loop (single one-step eval); its seams are entry +
                // pre-forward in SeedVR2ImageRefiner.refine.
                .init(phase: .upsample, unit: .chunk),
            ]))
        #expect(report.passed, "\(report.summary)")
    }
}
