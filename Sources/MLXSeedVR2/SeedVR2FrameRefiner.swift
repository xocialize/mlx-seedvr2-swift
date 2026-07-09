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
        let seedRef = seed, model = upscaler, doCC = colorCorrect
        return try tiler.process(upsized) { tile in
            // Cooperative cancellation once per diffusion tile (CAN mid-run cadence): sync code
            // on the run's task sees the flag; the CancellationError propagates unchanged
            // through the throwing tiler closure.
            try Task.checkCancellation()
            // tile: [1, th, tw, 3] NHWC RGB in [0,1] → [-1,1] NCHW (= style, the upscaled input)
            let style = tile.transposed(0, 3, 1, 2) * 2 - 1
            var refined = model.upscale(processedImage: style, seed: seedRef)   // [1,3,1,th,tw]
            if refined.ndim == 5 { refined = refined[0..., 0..., 0] }           // [1,3,th,tw]
            let corrected = doCC
                ? SeedVR2ColorCorrect.labTransfer(content: refined, style: style, luminanceWeight: 0.8)
                : refined
            return clip((corrected + 1) * 0.5, min: 0, max: 1).transposed(0, 2, 3, 1)  // [1,th,tw,3]
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
