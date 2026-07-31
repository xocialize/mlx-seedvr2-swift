//
//  SeedVR2TemporalRefiner.swift
//  MLXSeedVR2
//
//  The temporal chunking driver (GAP-PROGRAM V12-D) — the piece between the finished temporal
//  stack (V12-S streaming memory, N11 scene-cut detection) and the shipping `videoUpscale`
//  surface. Schedules decoded frames into T-frame windows, runs each window through the
//  temporal path (Lanczos pre-upscale → tiled 1:1 refine → the T>1 VAE), streams VAE state
//  across window joins, and ENDS A WINDOW EARLY at a detected scene cut (flush-only recovers
//  ~18 % of the cross-shot artefact; early termination recovers 100 % — n11_reset_at_cut.out).
//
//  Structure per window:
//    · every decoded source frame is Lanczos-upscaled immediately (the decoder recycles its
//      buffers, so nothing of the reader's is retained) and fed to the SHIPPING cut detector
//      (media-bridge `SceneCutDetector`, v0.11.0 defaults — recall-first upper-tail k=2.0);
//    · a full window (or a cut/end-of-clip tail, padded to the causal arithmetic's next legal
//      length) is refined tile position by tile position: each spatial tile is its own causal
//      stream, so each tile position carries its own `VAEStreamingBank`, adopted before and
//      exported after its chunk;
//    · one noise field per chunk is drawn over the whole frame's latent and sliced per tile —
//      continuous across tile seams (the stills v0.7.2 construction) while every steady-state
//      chunk still draws the byte-identical field (V12-S §4c: a per-chunk fixed-seed field is
//      the stable choice on the time axis; one long field is measurably worse);
//    · outputs are feather-blended with the same half-pixel-centred ramp and normalised
//      float32 composite as `MLXTileProcessor`'s `SeamAccumulator` (WORKORDER-tile-compositor),
//      colour-matched ONCE per frame against its own Lanczos base (the v0.7.4 construction),
//      and emitted with their recorded source PTS (frame-stream-native 0.4.0 timed transform).
//
//  T = 1 does not reach this type: `SeedVR2UpscalePackage.runVideo` routes it to the
//  pre-temporal per-frame path (`SeedVR2FrameRefiner`), bit-identical to v0.7.x.
//

import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import MediaMeasure
import MLX
import SeedVR2MLX
import UniformTypeIdentifiers

/// Streaming temporal driver over a loaded `SeedVR2Upscaler`. Feed `ingest` once per decoded
/// source frame in order, then `flush()` once; single-threaded per run (frames stream one at a
/// time through `NativeFrameStream.run`).
final class SeedVR2TemporalRefiner: @unchecked Sendable {
    private let upscaler: SeedVR2Upscaler
    private let tileSize: Int
    private let tileOverlap: Int
    private let colorCorrect: Bool
    private let seed: UInt64
    private let scale: Int
    private let sceneCutDetection: Bool

    private var planner: TemporalWindowPlanner
    private let detector = SceneCutDetector()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    /// Buffered window: each entry is OUR OWN Lanczos-upscaled BGRA buffer plus the source PTS.
    private var window: [(frame: CVPixelBuffer, pts: CMTime)] = []
    /// Per-tile-position streaming banks, in tile raster order (nil until a tile's first
    /// `.initializing` chunk has run). Rebuilt (as nils) on every stream flush.
    private var banks: [VAEStreamingBank?] = []
    /// Frame index of the next source frame (diagnostics).
    private var frameIndex = 0
    private var emitted = 0

    /// Diagnostics, receipt-driven: SEEDVR2_TEMPORAL_LOG=file appends one line per event
    /// (cut / chunk / flush); SEEDVR2_TEMPORAL_DUMP=dir writes each emitted frame as a PNG
    /// (f0000.png …) BEFORE the HEVC encode plus a chunks.txt of real chunk lengths — the
    /// format `probes/v12s_metrics.py` splits on.
    private let logURL = ProcessInfo.processInfo.environment["SEEDVR2_TEMPORAL_LOG"]
        .map { URL(fileURLWithPath: $0) }
    private let dump = SeedVR2TemporalDump()
    private var chunkLengths: [Int] = []

    init(upscaler: SeedVR2Upscaler, temporalWindow: Int, scale: Int, tileSize: Int,
         tileOverlap: Int, colorCorrect: Bool, seed: UInt64, sceneCutDetection: Bool = true) {
        precondition(temporalWindow >= 5 && temporalWindow % 4 == 1,
                     "temporal refiner needs a 4k+1 window ≥ 5; T=1 routes to the per-frame path")
        self.upscaler = upscaler
        self.planner = TemporalWindowPlanner(window: temporalWindow)
        self.scale = scale
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.colorCorrect = colorCorrect
        self.seed = seed
        self.sceneCutDetection = sceneCutDetection
    }

    // MARK: - Streaming entry points (the timed NativeFrameStream transform)

    /// One decoded source frame arrives. Returns zero or more finished output frames, each
    /// paired with its recorded source PTS.
    func ingest(_ source: CVPixelBuffer, pts: CMTime) throws -> [(CVPixelBuffer, CMTime)] {
        try Task.checkCancellation()

        // Detector first, on the frame already decoded — the cheap 40×24 luma-grid statistic,
        // BEFORE any model work. v0.11.0 defaults; a fired cut means THIS frame starts a new
        // shot, so the buffered window (the old shot) is closed before this frame joins.
        var isCut = false
        if sceneCutDetection, let decision = detector.next(source) {
            isCut = decision.isCut
            if isCut {
                log("cut frame=\(frameIndex) score=\(fmt(decision.score)) threshold=\(fmt(decision.threshold))")
            }
        }

        let upsized = try lanczosUpscale(source, factor: scale)
        let actions = planner.ingest(cutBeforeThisFrame: isCut)
        var out: [(CVPixelBuffer, CMTime)] = []

        if let early = actions.earlyChunk {
            out += try run(chunk: early)
        }
        if actions.resetStream {
            log("reset boundary-aligned cut frame=\(frameIndex)")
            flushStream()
        }
        window.append((upsized, pts))
        frameIndex += 1
        if let full = actions.fullChunk {
            out += try run(chunk: full)
        }
        return out
    }

    /// End of clip: run the buffered tail (padded to the next legal length, pad outputs
    /// trimmed) and release everything.
    func flush() throws -> [(CVPixelBuffer, CMTime)] {
        var out: [(CVPixelBuffer, CMTime)] = []
        if let tail = planner.flush() {
            out += try run(chunk: tail)
        }
        dump.finish(chunkLengths: chunkLengths)
        log("flush emitted=\(emitted)")
        return out
    }

    // MARK: - One chunk through the temporal path

    private func run(chunk: TemporalWindowPlanner.Chunk) throws -> [(CVPixelBuffer, CMTime)] {
        precondition(window.count == chunk.frameCount,
                     "planner scheduled \(chunk.frameCount) frames; driver holds \(window.count)")
        let frames = window
        window.removeAll(keepingCapacity: true)

        let width = CVPixelBufferGetWidth(frames[0].frame)
        let height = CVPixelBufferGetHeight(frames[0].frame)
        let grid = tileGrid(width: width, height: height)
        if banks.count != grid.count { banks = Array(repeating: nil, count: grid.count) }

        log("chunk frames=\(chunk.frameCount) padded=\(chunk.paddedCount) "
            + "state=\(chunk.memoryState == .active ? "active" : "init") "
            + "endsSegment=\(chunk.endsSegment) tiles=\(grid.count) at=\(frameIndex - chunk.frameCount)")

        // One noise field per chunk over the whole frame's latent, sliced per tile (header §3).
        precondition(tileSize % 16 == 0, "tileSize \(tileSize) must be /16")
        let tileLat = tileSize / 8
        let fieldH = max((height + 7) / 8, tileLat)
        let fieldW = max((width + 7) / 8, tileLat)
        let wholeNoise = SeedVR2LatentCreator.noiseLatents(
            seed: seed, height: fieldH, width: fieldW, latentFrames: chunk.latentFrames)
        eval(wholeNoise)

        let n = chunk.frameCount, padded = chunk.paddedCount
        // Feathered float32 composite per REAL frame (Σw·x and Σw, divide once — the
        // SeamAccumulator construction; weights are frame-invariant, so one weight plane).
        var colour = [Float](repeating: 0, count: n * height * width * 3)
        var weight = [Float](repeating: 0, count: height * width)

        for (tileIndex, tile) in grid.enumerated() {
            try Task.checkCancellation()

            // This tile position's own causal stream. `.initializing` reads no memory (cold
            // pad), but adopt anyway so a stale tail from another tile can never be read.
            upscaler.vae.adoptStreamingMemory(chunk.memoryState == .active ? banks[tileIndex] : nil)

            let stack = extractTileStack(frames: frames.map(\.frame), padded: padded, tile: tile,
                                         frameWidth: width, frameHeight: height)
            let ly = min(tile.y / 8, fieldH - tileLat), lx = min(tile.x / 8, fieldW - tileLat)
            let noise = wholeNoise[0..., 0..., 0..., ly ..< (ly + tileLat), lx ..< (lx + tileLat)]

            var refined = upscaler.upscale(processedImage: stack, noise: noise, seed: seed,
                                           numSteps: 1, memoryState: chunk.memoryState)
            // [1,3,T',ts,ts] in [-1,1] → [0,1]; trim the pad frames before readout.
            refined = clip((refined[0..., 0..., 0 ..< n] + 1) * 0.5, min: 0, max: 1)
            eval(refined)

            if chunk.endsSegment {
                banks[tileIndex] = nil
            } else {
                banks[tileIndex] = upscaler.vae.exportStreamingMemory()
            }

            accumulate(refined.asArray(Float.self), frames: n, tile: tile,
                       frameWidth: width, frameHeight: height,
                       colour: &colour, weight: &weight)
        }

        // ⚠️ Weight normalisation: `accumulate` adds each tile's ramp weight once per tile
        // (the weight plane is frame-invariant), and `resolveFrame` divides once — the
        // SeamAccumulator normalised-composite contract, corner-correct where four tiles meet.

        if chunk.endsSegment { flushStream() }

        // Resolve the composite per frame → BGRA (+ optional global LAB match vs its own base).
        var out: [(CVPixelBuffer, CMTime)] = []
        for f in 0 ..< n {
            try Task.checkCancellation()
            let buffer = try resolveFrame(f, colour: colour, weight: weight,
                                          width: width, height: height, base: frames[f].frame)
            dump.frame(buffer)
            emitted += 1
            out.append((buffer, frames[f].pts))
        }
        chunkLengths.append(n)
        return out
    }

    /// The stream ends (cut or clip end): drop every tile's bank and the VAE's live tails.
    private func flushStream() {
        banks = Array(repeating: nil, count: banks.count)
        upscaler.vae.resetStreamingMemory()
    }

    // MARK: - Tiling geometry (MLXTileProcessor's grid, deduplicated)

    private struct Tile { let x: Int, y: Int, w: Int, h: Int }

    /// Same clamped raster grid as `MLXTileProcessor.process` (step = tileSize − overlap,
    /// origins clamped so the last row/column stays in frame), with duplicate clamped origins
    /// collapsed — a duplicate tile is pure re-compute AND, here, a second streaming bank for
    /// the same pixels.
    private func tileGrid(width: Int, height: Int) -> [Tile] {
        let step = max(tileSize - tileOverlap, 1)
        var ys: [Int] = [], xs: [Int] = []
        for tileY in stride(from: 0, to: height, by: step) {
            let y = min(tileY, max(0, height - tileSize))
            if ys.last != y { ys.append(y) }
        }
        for tileX in stride(from: 0, to: width, by: step) {
            let x = min(tileX, max(0, width - tileSize))
            if xs.last != x { xs.append(x) }
        }
        var tiles: [Tile] = []
        for y in ys {
            for x in xs {
                tiles.append(Tile(x: x, y: y, w: min(tileSize, width - x),
                                  h: min(tileSize, height - y)))
            }
        }
        return tiles
    }

    /// Extract one tile position across the window into `[1,3,T',ts,ts]` in [−1,1] (BGRA →
    /// RGB, edge-replicated where the tile leaves the frame, pad frames = repeats of the last
    /// real frame).
    private func extractTileStack(frames: [CVPixelBuffer], padded: Int, tile: Tile,
                                  frameWidth: Int, frameHeight: Int) -> MLXArray {
        let ts = tileSize
        let plane = ts * ts
        var floats = [Float](repeating: 0, count: 3 * padded * plane)
        for t in 0 ..< padded {
            let buffer = frames[min(t, frames.count - 1)]
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(buffer)
            for ty in 0 ..< ts {
                let sy = min(max(tile.y + ty, 0), frameHeight - 1)
                for tx in 0 ..< ts {
                    let sx = min(max(tile.x + tx, 0), frameWidth - 1)
                    let p = sy * bpr + sx * 4                       // B,G,R,A
                    let o = t * plane + ty * ts + tx
                    floats[o] = Float(base[p + 2]) / 127.5 - 1                       // R
                    floats[plane * padded + o] = Float(base[p + 1]) / 127.5 - 1      // G
                    floats[2 * plane * padded + o] = Float(base[p]) / 127.5 - 1      // B
                }
            }
        }
        return MLXArray(floats, [1, 3, padded, ts, ts])
    }

    /// Half-pixel-centred feather ramp — `SeamAccumulator.ramp`, verbatim.
    @inline(__always)
    private static func ramp(_ i: Int, _ extent: Int, _ ov: Int) -> Float {
        let lo = (Float(i) + 0.5) / Float(ov)
        let hi = (Float(extent - 1 - i) + 0.5) / Float(ov)
        return min(min(lo, 1.0), min(hi, 1.0))
    }

    /// Feather-accumulate one refined tile stack (`[1,3,n,ts,ts]` flattened CHW-T floats in
    /// [0,1]) into the per-frame colour planes. The weight plane is frame-invariant (every
    /// frame gets the same tile geometry), so each tile adds its ramp weights ONCE — not once
    /// per frame — and `resolveFrame` divides every frame by the same Σw.
    private func accumulate(_ floats: [Float], frames n: Int, tile: Tile,
                            frameWidth: Int, frameHeight: Int,
                            colour: inout [Float], weight: inout [Float]) {
        let ts = tileSize
        let plane = ts * ts
        let ov = max(tileOverlap, 1)
        floats.withUnsafeBufferPointer { srcBuf in
            colour.withUnsafeMutableBufferPointer { colBuf in
                weight.withUnsafeMutableBufferPointer { wgtBuf in
                    let s = srcBuf.baseAddress!, c = colBuf.baseAddress!, w = wgtBuf.baseAddress!
                    for ty in 0 ..< min(tile.h, ts) {
                        let dstY = tile.y + ty
                        let wy = Self.ramp(ty, ts, ov)
                        for tx in 0 ..< min(tile.w, ts) {
                            let dstX = tile.x + tx
                            let wt = Self.ramp(tx, ts, ov) * wy
                            let dstP = dstY * frameWidth + dstX
                            w[dstP] += wt
                            for f in 0 ..< n {
                                let o = f * plane + ty * ts + tx
                                let dst = (f * frameHeight * frameWidth + dstP) * 3
                                c[dst + 0] += s[o] * wt                          // R
                                c[dst + 1] += s[plane * n + o] * wt              // G
                                c[dst + 2] += s[2 * plane * n + o] * wt          // B
                            }
                        }
                    }
                }
            }
        }
    }

    /// Divide the composite once, optionally LAB-match globally against the frame's own
    /// Lanczos base (the v0.7.4 per-frame construction), and write BGRA.
    private func resolveFrame(_ f: Int, colour: [Float], weight: [Float],
                              width: Int, height: Int, base: CVPixelBuffer) throws -> CVPixelBuffer {
        let plane = height * width
        var rgb = [Float](repeating: 0, count: 3 * plane)   // CHW for the tensor path
        colour.withUnsafeBufferPointer { colBuf in
            weight.withUnsafeBufferPointer { wgtBuf in
                let c = colBuf.baseAddress!, w = wgtBuf.baseAddress!
                for p in 0 ..< plane {
                    let sw = w[p]
                    let src = (f * plane + p) * 3
                    if sw > 0 {
                        rgb[p] = c[src] / sw
                        rgb[plane + p] = c[src + 1] / sw
                        rgb[2 * plane + p] = c[src + 2] / sw
                    }
                }
            }
        }

        let out = try makeBGRABuffer(width: width, height: height)
        var tensor = MLXArray(rgb, [1, 3, height, width])
        if colorCorrect {
            // Colour-match ONCE per frame against the whole pre-upscaled base — never per tile
            // (per-tile matching is the temporally unstable variant; SeedVR2FrameRefiner A/B).
            let style = try SeedVR2ImageRefiner.tensor(fromBGRA: base)
            let corrected = SeedVR2ColorCorrect.labTransfer(content: tensor * 2 - 1, style: style,
                                                            luminanceWeight: 0.8)
            tensor = clip((corrected + 1) * 0.5, min: 0, max: 1)
        }
        eval(tensor)
        try SeedVR2FrameRefiner.writeTensorToBGRA(tensor, into: out)
        return out
    }

    // MARK: - Small helpers

    private func makeBGRABuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var out: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &out) == kCVReturnSuccess,
              let buffer = out else {
            throw SeedVR2RefinerError.bufferAllocation("\(width)x\(height)")
        }
        return buffer
    }

    /// CoreImage Lanczos ×factor into OUR OWN buffer (the reader's buffers are recycled — the
    /// window must never retain one).
    private func lanczosUpscale(_ input: CVPixelBuffer, factor: Int) throws -> CVPixelBuffer {
        let w = CVPixelBufferGetWidth(input), h = CVPixelBufferGetHeight(input)
        let scaled = CIImage(cvPixelBuffer: input)
            .applyingFilter("CILanczosScaleTransform",
                            parameters: [kCIInputScaleKey: Double(factor), kCIInputAspectRatioKey: 1.0])
        let out = try makeBGRABuffer(width: w * factor, height: h * factor)
        ciContext.render(scaled, to: out)
        return out
    }

    private func log(_ line: String) {
        guard let logURL else { return }
        let data = (line + "\n").data(using: .utf8)!
        if let handle = try? FileHandle(forWritingTo: logURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: logURL)
        }
    }

    private func fmt(_ v: Double) -> String { String(format: "%.4f", v) }
}

/// Receipt diagnostic shared by the temporal driver and the T=1 per-frame path:
/// `SEEDVR2_TEMPORAL_DUMP=dir` writes every emitted output frame as `f%04d.png` BEFORE the
/// HEVC encode, plus a `chunks.txt` of real chunk lengths (all `1`s on the per-frame path) —
/// the format `probes/v12s_metrics.py` splits on. No-op when the variable is unset.
final class SeedVR2TemporalDump: @unchecked Sendable {
    private let dir = ProcessInfo.processInfo.environment["SEEDVR2_TEMPORAL_DUMP"]
        .map { URL(fileURLWithPath: $0, isDirectory: true) }
    private var count = 0

    init() {
        if let dir {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    func frame(_ buffer: CVPixelBuffer) {
        guard let dir else { return }
        let url = dir.appendingPathComponent(String(format: "f%04d.png", count))
        count += 1
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(data: base, width: CVPixelBufferGetWidth(buffer),
                                  height: CVPixelBufferGetHeight(buffer), bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info.rawValue),
              let cg = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                                         UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, cg, nil)
        CGImageDestinationFinalize(dest)
    }

    /// Write `chunks.txt`. The per-frame path passes no lengths → one `1` per emitted frame.
    func finish(chunkLengths: [Int] = []) {
        guard let dir else { return }
        let lens = chunkLengths.isEmpty ? Array(repeating: 1, count: count) : chunkLengths
        let txt = lens.map(String.init).joined(separator: "\n") + "\n"
        try? txt.write(to: dir.appendingPathComponent("chunks.txt"),
                       atomically: true, encoding: .utf8)
    }
}
