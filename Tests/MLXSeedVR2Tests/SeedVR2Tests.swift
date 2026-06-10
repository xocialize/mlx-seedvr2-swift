import Testing
import Foundation
import AVFoundation
import CoreVideo
import MLXToolKit
import FormatBridge
@testable import MLXSeedVR2

struct SeedVR2Tests {

    // MARK: - Offline conformance

    @Test func manifestIsVideoUpscaleAndPermissive() {
        let m = SeedVR2VideoUpscalePackage.manifest
        #expect(m.capabilities == [.videoUpscale])
        #expect(m.license.weightLicense == .apache2)
        #expect(m.license.portCodeLicense == .mit)
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func manifestRequirementsAreHeavy() {
        let r = SeedVR2VideoUpscalePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.chipFloor == .pro)
        #expect(r.footprints.contains { $0.quant == .int8 })
        #expect(r.footprints.contains { $0.quant == .bf16 })
    }

    @Test func surfaceIsTheCanonicalVideoUpscaleDescriptor() {
        let s = SeedVR2VideoUpscalePackage.manifest.surfaces.first
        #expect(s?.capability == .videoUpscale)
        #expect(s?.parameters.first?.kind == .video)
        #expect(s?.parameters.contains { $0.name == "scale" && !$0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = SeedVR2VideoUpscalePackage.registration
        #expect(reg.manifest.capabilities == [.videoUpscale])
        let pkg = try reg.makePackage(SeedVR2Configuration())
        #expect(pkg is SeedVR2VideoUpscalePackage)
    }

    @Test func configurationDefaultsAndCodable() throws {
        let c = SeedVR2Configuration()
        #expect(c.repo == "mlx-community/SeedVR2-3B-mlx-int8")
        #expect(c.defaultScale == 2)
        #expect(c.colorCorrect)

        var custom = SeedVR2Configuration(repo: "mlx-community/SeedVR2-3B-mlx", defaultScale: 4)
        custom.modelsRootDirectory = URL(fileURLWithPath: "/tmp/x")
        let back = try JSONDecoder().decode(SeedVR2Configuration.self, from: JSONEncoder().encode(custom))
        #expect(back.repo == "mlx-community/SeedVR2-3B-mlx")
        #expect(back.defaultScale == 4)
        #expect(back.modelsRootDirectory == nil)
    }

    // MARK: - Live frame-stream round-trip (format-bridge Layer-2; no Metal, runs in CLI)

    @Test func videoIOTranscodesWithPerFrameTransform() async throws {
        // Write a tiny 8-frame 64×64 HEVC video, then transcode it through a passthrough-double
        // transform (each frame replaced by a 2× buffer) and verify the output dimensions + frames.
        let tmp = FileManager.default.temporaryDirectory
        let src = tmp.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        let dst = tmp.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        defer { try? FileManager.default.removeItem(at: src); try? FileManager.default.removeItem(at: dst) }

        try await Self.writeTestVideo(to: src, width: 64, height: 64, frames: 8, fps: 8)

        let meta = try await FrameStreamTransform.run(input: src, output: dst) { frame in
            [try Self.scaledCopy(frame, factor: 2)]
        }
        #expect(meta.sourceWidth == 64)
        #expect(meta.sourceFrameRate > 0)
        #expect(meta.frameCount == 8)

        // Verify the written output: 128×128, ~8 frames, video track present.
        let outAsset = AVURLAsset(url: dst)
        let track = try #require(try await outAsset.loadTracks(withMediaType: .video).first)
        let outSize = try await track.load(.naturalSize)
        #expect(Int(outSize.width) == 128)
        #expect(Int(outSize.height) == 128)
    }

    // MARK: - Helpers

    enum TestVideoError: Error { case helper }

    static func writeTestVideo(to url: URL, width: Int, height: Int, frames: Int, fps: Int) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        for i in 0..<frames {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            guard let pool = adaptor.pixelBufferPool else { throw TestVideoError.helper }
            var pb: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
            guard let buffer = pb else { throw TestVideoError.helper }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                memset(base, Int32(20 + i * 25), CVPixelBufferGetDataSize(buffer))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps)))
        }
        input.markAsFinished()
        await writer.finishWriting()
    }

    /// Allocate a factor× BGRA buffer and block-copy (nearest) — a Metal-free stand-in transform.
    static func scaledCopy(_ src: CVPixelBuffer, factor: Int) throws -> CVPixelBuffer {
        let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
        let (ow, oh) = (w * factor, h * factor)
        var out: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: ow, kCVPixelBufferHeightKey as String: oh,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, ow, oh, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out) == kCVReturnSuccess,
              let dst = out else { throw TestVideoError.helper }
        CVPixelBufferLockBaseAddress(src, .readOnly); CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly); CVPixelBufferUnlockBaseAddress(dst, []) }
        guard let sBase = CVPixelBufferGetBaseAddress(src)?.assumingMemoryBound(to: UInt8.self),
              let dBase = CVPixelBufferGetBaseAddress(dst)?.assumingMemoryBound(to: UInt8.self) else {
            throw TestVideoError.helper
        }
        let sRow = CVPixelBufferGetBytesPerRow(src), dRow = CVPixelBufferGetBytesPerRow(dst)
        for y in 0..<oh {
            let sy = y / factor
            for x in 0..<ow {
                let sx = x / factor
                for c in 0..<4 { dBase[y * dRow + x * 4 + c] = sBase[sy * sRow + sx * 4 + c] }
            }
        }
        return dst
    }
}
