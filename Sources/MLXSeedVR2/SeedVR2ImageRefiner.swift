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
import CoreVideo
import Foundation
import ImageIO
import MLX
import MLXToolKit
import RealESRGANMLX   // MLXTileProcessor — the shared feathered-seam tiler
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
    private let tileSize: Int
    private let tileOverlap: Int
    private let wholeFramePixels: Int
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    init(upscaler: SeedVR2Upscaler, seed: UInt64, colorCorrect: Bool,
         tileSize: Int = 256, tileOverlap: Int = 32, wholeFramePixels: Int = 1024 * 1024) {
        self.upscaler = upscaler
        self.seed = seed
        self.colorCorrect = colorCorrect
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.wholeFramePixels = wholeFramePixels
    }

    /// Upscale one image by `factor` (2 or 4): Lanczos pre-upscale → 1:1 diffusion refine → crop.
    func refine(_ image: Image, factor: Int) throws -> Image {
        guard factor == 2 || factor == 4 else { throw SeedVR2ImageRefinerError.unsupportedScale(factor) }

        let src = try Self.decodeCGImage(image)
        let upsized = try lanczosUpscale(src, factor: factor)         // exact (w·f, h·f)
        let outW = upsized.width, outH = upsized.height

        // Above the measured envelope, refine in tiles. The diffusion still runs at 1:1 — only the
        // *extent* of each call changes — so this is the same operation, kept inside the model's regime.
        if wholeFramePixels == 0 || outW * outH > wholeFramePixels {
            return try refineTiled(upsized, width: outW, height: outH)
        }

        // [1,3,Hpad,Wpad] in [-1,1], edge-replicated to the next /16 (VAE 8× + patch/window need it).
        let style = try Self.tensorFromCGImage(upsized)

        // Pre-forward cancellation checkpoint (CAN): the V1 image path is a single monolithic
        // one-step diffusion eval (no tile loop), so the real seams are entry + here.
        try Task.checkCancellation()
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

    // MARK: - Tiled refine (above the envelope)

    /// Refine an already-upscaled image tile-by-tile at 1:1 with feathered seams.
    ///
    /// ⚠️ The closure below is a deliberate mirror of `SeedVR2FrameRefiner.refine`'s — same NHWC→NCHW
    /// transpose, same `[0,1]`→`[-1,1]` mapping, same LAB transfer with `luminanceWeight: 0.8`, same clip
    /// on the way out. If one is ever changed, change both, or the two surfaces stop agreeing on what
    /// "refined" means.
    private func refineTiled(_ upsized: CGImage, width: Int, height: Int) throws -> Image {
        let buffer = try Self.pixelBuffer(from: upsized)
        let tiler = MLXTileProcessor(tileSize: tileSize, overlap: tileOverlap, scale: 1)
        let model = upscaler

        // 🔑 **ONE noise field over the whole image's latent, sliced per tile — the correct
        // construction.** `SeedVR2LatentCreator.noiseLatents` sized per call would give each tile its
        // own independent field: identical everywhere with one seed (periodic texture locked to the
        // tile grid — shipped v0.7.2 decorrelated that with a per-tile seed), and DISCONTINUOUS across
        // seams either way. Drawing the field once at the image's latent resolution (1/8 pixel for the
        // 8× VAE) and slicing each tile's region keeps the field continuous across seams, matching the
        // single-pass branch's semantics. The slice origin needs the tile's position, which the
        // region-aware `MLXTileProcessor.process` overload (mlx-realesrgan-swift 0.6.0) provides.
        //
        // ⚠️ Latent cells are 8 px; a clamped last-row/column tile origin may not be 8-aligned, so its
        // slice can sit up to 7 px off the tile's true position. Sub-cell, edge-tiles-only — accepted.
        // The field is padded up to the tile latent so a tile larger than the image (skinny inputs)
        // still slices in bounds; the excess only ever lands under edge-replicated padding pixels.
        precondition(tileSize % 16 == 0, "tileSize \(tileSize) must be /16 (VAE 8× + patch/window)")
        let tileLat = tileSize / 8
        let fieldH = max((height + 7) / 8, tileLat), fieldW = max((width + 7) / 8, tileLat)
        let wholeNoise = SeedVR2LatentCreator.noiseLatents(seed: seed, height: fieldH, width: fieldW)
        eval(wholeNoise)

        let out = try tiler.process(buffer) { tile, region in
            // Per-tile cancellation: with tiling there IS a mid-run seam, unlike the single-pass branch.
            try Task.checkCancellation()
            let ly = min(region.y / 8, fieldH - tileLat), lx = min(region.x / 8, fieldW - tileLat)
            let noise = wholeNoise[0..., 0..., 0..., ly ..< (ly + tileLat), lx ..< (lx + tileLat)]
            let style = tile.transposed(0, 3, 1, 2) * 2 - 1              // [1,3,th,tw] in [-1,1]
            var refined = model.upscale(processedImage: style, noise: noise)
            if refined.ndim == 5 { refined = refined[0..., 0..., 0] }
            // 🚨 **NO per-tile colour transfer — see `colorCorrect` below.**
            let result = clip((refined + 1) * 0.5, min: 0, max: 1).transposed(0, 2, 3, 1)
            eval(result)
            return result
        }

        // 🚨 **The colour transfer runs ONCE, on the assembled image, and this is the whole reason the
        // tiled branch is not just the frame path with a different input type.**
        //
        // `labTransfer` is `histMatch` on the LAB a/b channels (and partially on L) — a **global
        // statistics** operation. Applied per tile, every 256² tile is matched to *its own* local
        // histogram, so each one lands in a slightly different colour space: measured as a visible
        // tile-grid across flat regions plus chroma speckle where a local a/b histogram is nearly
        // degenerate (dark, near-neutral areas). Overlap feathering cannot repair that — it blends
        // neighbours that disagree about what neutral *is*.
        //
        // Matching once against the whole pre-upscaled base restores the single-pass semantics exactly,
        // which is also why tiled and untiled output now agree instead of being two different looks.
        //
        // ⚠️ **`SeedVR2FrameRefiner` still colour-matches per tile and therefore still has this defect.**
        // It is not fixed here on purpose: video needs its own validation pass (temporal stability of a
        // per-frame global match is a real question), and silently changing the validated export tier
        // while chasing a stills bug is how two surfaces start disagreeing again.
        if colorCorrect {
            let assembled = try Self.tensor(fromBGRA: out)                    // [1,3,H,W] in [-1,1]
            let style = try Self.tensorFromCGImage(upsized)[0..., 0..., 0 ..< height, 0 ..< width]
            let corrected = SeedVR2ColorCorrect.labTransfer(content: assembled, style: style,
                                                            luminanceWeight: 0.8)
            let mapped = clip((corrected + 1) * 0.5, min: 0, max: 1)
            eval(mapped)
            let data = try Self.pngFromTensor(mapped, width: width, height: height)
            return Image(format: .png, data: data, width: width, height: height)
        }
        return try Self.image(fromBGRA: out, width: width, height: height)
    }

    /// BGRA `CVPixelBuffer` → `[1,3,H,W]` float tensor in `[-1,1]`, the currency `labTransfer` takes.
    static func tensor(fromBGRA buffer: CVPixelBuffer) throws -> MLXArray {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw SeedVR2ImageRefinerError.pixelAccessFailed
        }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        var floats = [Float](repeating: 0, count: 3 * h * w)
        let plane = h * w
        for y in 0 ..< h {
            for x in 0 ..< w {
                let p = y * stride + x * 4      // BGRA byte order: B,G,R,A
                let o = y * w + x
                floats[o]             = Float(bytes[p + 2]) / 127.5 - 1   // R
                floats[plane + o]     = Float(bytes[p + 1]) / 127.5 - 1   // G
                floats[2 * plane + o] = Float(bytes[p]) / 127.5 - 1       // B
            }
        }
        return MLXArray(floats, [1, 3, h, w])
    }

    /// CGImage → BGRA `CVPixelBuffer`, the currency `MLXTileProcessor` takes.
    static func pixelBuffer(from cg: CGImage) throws -> CVPixelBuffer {
        let w = cg.width, h = cg.height
        var out: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &out) == kCVReturnSuccess,
              let buffer = out else { throw SeedVR2ImageRefinerError.pixelAccessFailed }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        // ⚠️ BGRA byte order = `.byteOrder32Little` + `premultipliedFirst`; getting this wrong swaps the
        // red and blue channels, which reads as a colour-graded result rather than as a bug.
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: w, height: h,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info.rawValue)
        else { throw SeedVR2ImageRefinerError.pixelAccessFailed }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// BGRA `CVPixelBuffer` → PNG `Image`, matching the single-pass branch's return contract.
    static func image(fromBGRA buffer: CVPixelBuffer, width: Int, height: Int) throws -> Image {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(data: base, width: CVPixelBufferGetWidth(buffer),
                                  height: CVPixelBufferGetHeight(buffer), bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info.rawValue),
              let cg = ctx.makeImage() else { throw SeedVR2ImageRefinerError.encodeFailed }

        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw SeedVR2ImageRefinerError.encodeFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw SeedVR2ImageRefinerError.encodeFailed }
        return Image(format: .png, data: data as Data, width: width, height: height)
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
