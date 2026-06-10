//
//  VideoIO.swift
//  MLXSeedVR2
//
//  Streaming video decode/encode for the Video→Video transform path.
//  Forge conventions carried over:
//  - Always tag the encoder BT.709 (primaries/transfer/matrix) — an untagged stream makes
//    players guess and a 601-vs-709 mismatch drifts saturated colors (forge #61).
//  - Frames stream one at a time (decode → transform → append) so memory stays bounded.
//

import AVFoundation
import CoreVideo
import Foundation

public enum VideoIOError: Error {
    case openFailed(String)
    case noVideoTrack
    case readFailed(String)
    case writeFailed(String)
}

enum VideoIO {

    struct Metadata {
        let width: Int
        let height: Int
        let frameRate: Double
        let duration: Double
    }

    /// Stream-process a video file: decode each frame (BGRA), apply `transform`, encode HEVC.
    /// Returns the output metadata. `transform` runs on the caller's actor; cancellation is
    /// checked per frame.
    static func transcode(
        input: URL,
        output: URL,
        transform: (CVPixelBuffer) async throws -> CVPixelBuffer
    ) async throws -> Metadata {
        let asset = AVURLAsset(url: input)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoIOError.noVideoTrack
        }
        let size = try await track.load(.naturalSize)
        let fps = try await track.load(.nominalFrameRate)
        let duration = try await asset.load(.duration).seconds

        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        reader.add(readerOutput)
        guard reader.startReading() else {
            throw VideoIOError.readFailed(reader.error?.localizedDescription ?? "startReading")
        }

        // Output dimensions come from the first transformed frame.
        var writer: AVAssetWriter?
        var writerInput: AVAssetWriterInput?
        var adaptor: AVAssetWriterInputPixelBufferAdaptor?

        var frameIndex = 0
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(max(fps, 1)))

        while let sample = readerOutput.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let inPB = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)

            let outPB = try await transform(inPB)

            if writer == nil {
                let ow = CVPixelBufferGetWidth(outPB), oh = CVPixelBufferGetHeight(outPB)
                let w = try AVAssetWriter(outputURL: output, fileType: .mp4)
                let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.hevc,
                    AVVideoWidthKey: ow,
                    AVVideoHeightKey: oh,
                    // BT.709, always tagged (forge #61).
                    AVVideoColorPropertiesKey: [
                        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
                    ],
                ])
                input.expectsMediaDataInRealTime = false
                let a = AVAssetWriterInputPixelBufferAdaptor(
                    assetWriterInput: input,
                    sourcePixelBufferAttributes: [
                        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                        kCVPixelBufferWidthKey as String: ow,
                        kCVPixelBufferHeightKey as String: oh,
                    ])
                w.add(input)
                guard w.startWriting() else {
                    throw VideoIOError.writeFailed(w.error?.localizedDescription ?? "startWriting")
                }
                w.startSession(atSourceTime: .zero)
                writer = w; writerInput = input; adaptor = a
            }

            guard let input = writerInput, let adaptor else { break }
            // Video-only track: a bounded wait for readiness is safe (no cross-track interleave).
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 2_000_000)
                try Task.checkCancellation()
            }
            let t = pts.isValid && pts.isNumeric ? pts
                : CMTimeMultiply(frameDuration, multiplier: Int32(frameIndex))
            guard adaptor.append(outPB, withPresentationTime: t) else {
                throw VideoIOError.writeFailed(writer?.error?.localizedDescription ?? "append frame \(frameIndex)")
            }
            frameIndex += 1
        }

        if reader.status == .failed {
            throw VideoIOError.readFailed(reader.error?.localizedDescription ?? "reader failed")
        }
        guard let writer, let writerInput else {
            throw VideoIOError.readFailed("no frames decoded")
        }
        writerInput.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed {
            throw VideoIOError.writeFailed(writer.error?.localizedDescription ?? "finishWriting")
        }

        return Metadata(width: Int(size.width), height: Int(size.height),
                        frameRate: Double(fps), duration: duration)
    }
}
