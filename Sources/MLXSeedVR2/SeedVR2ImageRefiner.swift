//
//  SeedVR2ImageRefiner.swift
//  MLXSeedVR2
//
//  The host-side half of SeedVR2 image super-resolution (the model itself is 1:1 — it refines,
//  it does not resize). Per image:
//    1. CoreImage Lanczos pre-upscale by the integer factor (the spatial 2×/4×).
//    2. SeedVR2 one-step diffusion REFINES the upscaled image at 1:1 (edge-padded to /16 for the
//       8× VAE + patch/window, then cropped back).
//    3. Optional LAB-wavelet color transfer of the refined detail toward the upscaled base.
//
//  V1 is single-pass (no tiling): activation is resolution-driven, so the footprint is measured
//  and declared at a documented input envelope. Tiling for very large images is a future lever
//  (the core delegates it to the host; cf. ForgeUpscaler's MLXTileProcessor).
//

import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import MLXToolKit
import SeedVR2MLX
import UniformTypeIdentifiers

public enum SeedVR2ImageRefinerError: Error {
    case unsupportedScale(Int)
    case decodeFailed
    case pixelAccessFailed
    case encodeFailed
}

/// Stateless per-call image refiner over a loaded `SeedVR2Upscaler`.
final class SeedVR2ImageRefiner: @unchecked Sendable {
    private let upscaler: SeedVR2Upscaler
    private let seed: UInt64
    private let colorCorrect: Bool
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    init(upscaler: SeedVR2Upscaler, seed: UInt64, colorCorrect: Bool) {
        self.upscaler = upscaler
        self.seed = seed
        self.colorCorrect = colorCorrect
    }

    /// Upscale one image by `factor` (2 or 4): Lanczos pre-upscale → 1:1 diffusion refine → crop.
    func refine(_ image: Image, factor: Int) throws -> Image {
        guard factor == 2 || factor == 4 else { throw SeedVR2ImageRefinerError.unsupportedScale(factor) }

        let src = try Self.decodeCGImage(image)
        let upsized = try lanczosUpscale(src, factor: factor)         // exact (w·f, h·f)
        let outW = upsized.width, outH = upsized.height

        // [1,3,Hpad,Wpad] in [-1,1], edge-replicated to the next /16 (VAE 8× + patch/window need it).
        let style = try Self.tensorFromCGImage(upsized)

        var refined = upscaler.upscale(processedImage: style, seed: seed)  // [1,3,1,Hpad,Wpad], [-1,1]
        if refined.ndim == 5 { refined = refined[0..., 0..., 0] }          // → [1,3,Hpad,Wpad]
        if colorCorrect {
            refined = SeedVR2ColorCorrect.labTransfer(content: refined, style: style, luminanceWeight: 0.8)
        }
        // Crop the /16 pad back to the true upscaled size, map [-1,1] → [0,1].
        let cropped = clip((refined[0..., 0..., 0 ..< outH, 0 ..< outW] + 1) * 0.5, min: 0, max: 1)
        eval(cropped)

        let data = try Self.pngFromTensor(cropped, width: outW, height: outH)
        return Image(format: .png, data: data, width: outW, height: outH)
    }

    // MARK: - CoreImage Lanczos

    private func lanczosUpscale(_ cg: CGImage, factor: Int) throws -> CGImage {
        let (ow, oh) = (cg.width * factor, cg.height * factor)
        let scaled = CIImage(cgImage: cg).applyingFilter("CILanczosScaleTransform",
            parameters: [kCIInputScaleKey: Double(factor), kCIInputAspectRatioKey: 1.0])
        guard let out = ciContext.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: ow, height: oh)) else {
            throw SeedVR2ImageRefinerError.decodeFailed
        }
        return out
    }

    // MARK: - CGImage ⇆ MLXArray (the silent-failure-prone seam — channel order + range are explicit)

    static func decodeCGImage(_ image: Image) throws -> CGImage {
        switch image.format {
        case .png, .jpeg:
            guard let srcRef = CGImageSourceCreateWithData(image.data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(srcRef, 0, nil) else {
                throw SeedVR2ImageRefinerError.decodeFailed
            }
            return cg
        case .rawBGRA8:
            guard let w = image.width, let h = image.height else { throw SeedVR2ImageRefinerError.decodeFailed }
            let cs = CGColorSpaceCreateDeviceRGB()
            let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)   // BGRA8
            guard let provider = CGDataProvider(data: image.data as CFData),
                  let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                                   bytesPerRow: w * 4, space: cs, bitmapInfo: info, provider: provider,
                                   decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
                throw SeedVR2ImageRefinerError.decodeFailed
            }
            return cg
        }
    }

    /// Render the CGImage into a known RGBA8 buffer, then pack a [1,3,Hpad,Wpad] float tensor in
    /// [-1,1], edge-replicating the real-content border out to the next multiple of 16.
    static func tensorFromCGImage(_ cg: CGImage) throws -> MLXArray {
        let w = cg.width, h = cg.height
        let wPad = roundUp(w, 16), hPad = roundUp(h, 16)
        let bytesPerRow = w * 4
        var rgba = [UInt8](repeating: 0, count: h * bytesPerRow)
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue   // RGBA8, byte order R,G,B,A
        guard let ctx = rgba.withUnsafeMutableBytes({ ptr in
            CGContext(data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: bytesPerRow, space: cs, bitmapInfo: info)
        }) else { throw SeedVR2ImageRefinerError.pixelAccessFailed }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        // CHW float, normalized to [-1,1]; pad indices clamp to the edge (replicate, not black).
        var floats = [Float](repeating: 0, count: 3 * hPad * wPad)
        let plane = hPad * wPad
        for y in 0 ..< hPad {
            let sy = min(y, h - 1)
            for x in 0 ..< wPad {
                let sx = min(x, w - 1)
                let p = sy * bytesPerRow + sx * 4
                let r = Float(rgba[p]) / 127.5 - 1
                let g = Float(rgba[p + 1]) / 127.5 - 1
                let b = Float(rgba[p + 2]) / 127.5 - 1
                let o = y * wPad + x
                floats[o] = r
                floats[plane + o] = g
                floats[2 * plane + o] = b
            }
        }
        return MLXArray(floats, [1, 3, hPad, wPad])
    }

    /// [1,3,H,W] in [0,1] → RGBA8 PNG of size (width,height).
    static func pngFromTensor(_ t: MLXArray, width w: Int, height h: Int) throws -> Data {
        let flat = (t * 255).asType(.uint8).asArray(UInt8.self)   // CHW, row-major
        let plane = h * w
        var rgba = [UInt8](repeating: 255, count: h * w * 4)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let i = y * w + x
                let o = i * 4
                rgba[o] = flat[i]
                rgba[o + 1] = flat[plane + i]
                rgba[o + 2] = flat[2 * plane + i]
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = rgba.withUnsafeMutableBytes({ ptr in
            CGContext(data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: w * 4, space: cs, bitmapInfo: info)
        }), let cg = ctx.makeImage() else { throw SeedVR2ImageRefinerError.encodeFailed }

        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else {
            throw SeedVR2ImageRefinerError.encodeFailed
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw SeedVR2ImageRefinerError.encodeFailed }
        return out as Data
    }

    private static func roundUp(_ v: Int, _ m: Int) -> Int { ((v + m - 1) / m) * m }
}
