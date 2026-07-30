//
//  SeedVR2FrameRefiner.swift
//  MLXSeedVR2
//
//  Per-frame refinement, ported from forge-studio-optimizer's validated SeedVR2_MLX
//  export tier (runtime e2e GPU/int8 + W4 LAB color-correct, Forge PR #2):
//    1. CoreImage Lanczos pre-upscale by the integer factor (the spatial SR).
//    2. SeedVR2 one-step diffusion REFINES the upscaled frame at 1:1, tile-blended
//       (the shared MLXTileProcessor with scale = 1 — feathered seams).
//    3. LAB-wavelet color transfer of the refined detail toward the upscaled base.
//

import CoreImage
import CoreVideo
import Foundation
import MLX
import RealESRGANMLX
import SeedVR2MLX

/// Errors in the per-frame refinement path.
public enum SeedVR2RefinerError: Error {
    case unsupportedScale(Int)
    case bufferAllocation(String)
}

final class SeedVR2FrameRefiner: @unchecked Sendable {
    private let upscaler: SeedVR2Upscaler
    private let tileSize: Int
    private let tileOverlap: Int
    private let colorCorrect: Bool
    private let seed: UInt64
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    init(upscaler: SeedVR2Upscaler, tileSize: Int, tileOverlap: Int, colorCorrect: Bool, seed: UInt64) {
        self.upscaler = upscaler
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.colorCorrect = colorCorrect
        self.seed = seed
    }

    /// Upscale one BGRA frame by `factor` (2 or 4).
    func refine(_ buffer: CVPixelBuffer, factor: Int) throws -> CVPixelBuffer {
        guard factor == 2 || factor == 4 else { throw SeedVR2RefinerError.unsupportedScale(factor) }

        let upsized = try lanczosUpscale(buffer, factor: factor)

        let tiler = MLXTileProcessor(tileSize: tileSize, overlap: tileOverlap, scale: 1)
        let seedRef = seed, model = upscaler
        let out = try tiler.process(upsized) { tile in
            // Cooperative cancellation once per diffusion tile (CAN mid-run cadence): sync code
            // on the run's task sees the flag; the CancellationError propagates unchanged
            // through the throwing tiler closure.
            try Task.checkCancellation()
            // tile: [1, th, tw, 3] NHWC RGB in [0,1] → [-1,1] NCHW (= style, the upscaled input)
            let style = tile.transposed(0, 3, 1, 2) * 2 - 1
            var refined = model.upscale(processedImage: style, seed: seedRef)   // [1,3,1,th,tw]
            if refined.ndim == 5 { refined = refined[0..., 0..., 0] }           // [1,3,th,tw]
            // 🚨 NO per-tile colour transfer — the frame is matched ONCE, globally, below.
            return clip((refined + 1) * 0.5, min: 0, max: 1).transposed(0, 2, 3, 1)  // [1,th,tw,3]
        }

        // 🔑 **Colour-match ONCE per frame, against the whole pre-upscaled base — the stills v0.7.1
        // construction, ported after the temporal question was answered by measurement (V10-fix,
        // 2026-07-30).** `labTransfer` is `histMatch` on global statistics; per tile it puts every
        // tile in its own colour space (visible tile grid + chroma speckle in flats). The video path
        // had deliberately kept the per-tile match pending a flicker A/B — the worry being that a
        // per-frame global match would make the whole frame's tone breathe with content. Measured on
        // a 64-frame 2 px/frame pan (256→512 ×2, 256/32 tiles) against a Lanczos-only content floor
        // of mean|Δa*| 0.110 / max 0.269 per frame-pair:
        //   per-tile match:  mean|Δa*| 0.164, max 1.62  (spikes 6× the content floor — a tile's
        //                    local histogram churns as content crosses its border, and the whole
        //                    tile's colour jumps with it)
        //   global match:    mean|Δa*| 0.092, max 0.235 (BELOW the content floor on both stats)
        // The feared direction inverted: the per-tile match was the temporally unstable one, and the
        // global match is more stable than the interpolation baseline on top of being spatially
        // correct. One clip / one content class / one geometry — but the margin is not close.
        if colorCorrect {
            let assembled = try SeedVR2ImageRefiner.tensor(fromBGRA: out)      // [1,3,H,W] in [-1,1]
            let style = try SeedVR2ImageRefiner.tensor(fromBGRA: upsized)
            let corrected = SeedVR2ColorCorrect.labTransfer(content: assembled, style: style,
                                                            luminanceWeight: 0.8)
            let mapped = clip((corrected + 1) * 0.5, min: 0, max: 1)
            eval(mapped)
            try Self.writeTensorToBGRA(mapped, into: out)
            return out
        }
        return out
    }

    /// Write a `[1,3,H,W]` tensor in `[0,1]` back into an existing BGRA buffer of the same size.
    static func writeTensorToBGRA(_ t: MLXArray, into buffer: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        guard t.shape.count == 4, t.shape[2] == h, t.shape[3] == w,
              let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw SeedVR2RefinerError.bufferAllocation("tensor \(t.shape) vs buffer \(w)x\(h)")
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let flat = (t * 255).asType(.uint8).asArray(UInt8.self)   // CHW, row-major
        let plane = h * w
        for y in 0 ..< h {
            for x in 0 ..< w {
                let o = y * w + x
                let p = y * stride + x * 4                        // BGRA
                bytes[p]     = flat[2 * plane + o]                // B
                bytes[p + 1] = flat[plane + o]                    // G
                bytes[p + 2] = flat[o]                            // R
                bytes[p + 3] = 255
            }
        }
    }

    /// CoreImage Lanczos upscale of a CVPixelBuffer by an integer factor.
    private func lanczosUpscale(_ input: CVPixelBuffer, factor: Int) throws -> CVPixelBuffer {
        let w = CVPixelBufferGetWidth(input), h = CVPixelBufferGetHeight(input)
        let (ow, oh) = (w * factor, h * factor)
        let ci = CIImage(cvPixelBuffer: input)
        let scaled = ci.applyingFilter("CILanczosScaleTransform",
                                       parameters: [kCIInputScaleKey: Double(factor),
                                                    kCIInputAspectRatioKey: 1.0])
        var out: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: ow, kCVPixelBufferHeightKey as String: oh,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, ow, oh, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out) == kCVReturnSuccess,
              let outBuffer = out else {
            throw SeedVR2RefinerError.bufferAllocation("\(ow)x\(oh)")
        }
        ciContext.render(scaled, to: outBuffer)
        return outBuffer
    }
}
