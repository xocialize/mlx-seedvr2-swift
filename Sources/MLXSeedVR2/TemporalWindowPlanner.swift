//
//  TemporalWindowPlanner.swift
//  MLXSeedVR2
//
//  Pure scheduling state machine for the temporal chunking driver (GAP-PROGRAM V12-D).
//  Decides — from frame arrivals, detector cut verdicts, and end-of-clip — which frames form
//  each VAE chunk, with which memory state, and how a ragged segment tail is padded. No pixels,
//  no MLX arrays: everything here is weightless-testable.
//
//  The causal arithmetic it encodes (V12-F/V12-S, ByteDance `slicing_encode`/`slicing_decode`):
//    · the FIRST chunk of a segment is `.initializing` and must be T ≡ 1 (mod 4) frames
//      (latT = 1 + (T−1)/4 — it carries the encoder's first-frame special case);
//    · every LATER chunk is `.active` and must be a multiple of 4 (latT = T/4);
//    · the two compose exactly, so a window knob of W (4k+1) schedules W, W−1, W−1, … frames.
//
//  Cut handling (N11, `probes/n11_reset_at_cut.out`): a detected cut ENDS THE CURRENT WINDOW
//  EARLY — the buffered frames run as a segment-final chunk, the stream is flushed, and the
//  cut frame starts a fresh `.initializing` segment. Flushing only at the next join recovers
//  ~18 % of the cross-shot artefact and leaves the straddling window's frames bit-identical
//  to no-reset; early termination recovers 100 %.
//
//  The 4k+1 rule does not close under splitting, so a segment-final chunk is PADDED up to the
//  next legal length by repeating the last real frame, and the pad outputs are trimmed. A pad
//  frame is only ever fed on a chunk whose stream is flushed right after — never forward into
//  another chunk's memory.
//

import Foundation
import SeedVR2MLX

/// See the header above. Feed `ingest(cutBeforeThisFrame:)` once per decoded frame in order,
/// then `flush()` once at end of clip; run the returned chunks over the frames the driver has
/// buffered. Total `frameCount` over all returned chunks always equals the frames ingested.
struct TemporalWindowPlanner {
    /// One VAE chunk to run over the oldest `frameCount` buffered frames.
    struct Chunk: Equatable {
        /// Real frames in the chunk (the driver consumes this many from its buffer).
        let frameCount: Int
        /// Legal length actually fed to the VAE (≥ `frameCount`; the difference is repeats of
        /// the last real frame, and their outputs are trimmed).
        let paddedCount: Int
        /// `.initializing` for a segment's first chunk, `.active` after.
        let memoryState: VAEMemoryState
        /// When true the driver must flush the stream (reset all banks) after running this
        /// chunk — the segment ended here (cut or end of clip).
        let endsSegment: Bool

        /// Latent frame count for this chunk (noise sizing).
        var latentFrames: Int {
            memoryState == .active ? paddedCount / 4 : 1 + (paddedCount - 1) / 4
        }
    }

    /// What the driver must do on one frame arrival, in order.
    struct Actions: Equatable {
        /// Run the ENTIRE buffer as a segment-final chunk BEFORE this frame joins it (a cut
        /// fired: the new frame belongs to the next shot). Implies a stream flush.
        var earlyChunk: Chunk?
        /// A cut fired exactly on a window boundary (nothing buffered): there is no chunk to
        /// run, but the stream still carries the previous shot's tail — flush it.
        var resetStream = false
        /// The buffer (with this frame appended) reached the window target: run it as an
        /// interior chunk, stream continues.
        var fullChunk: Chunk?
    }

    /// Largest legal window (1 or 4k+1) ≤ `requested`. 9 → 9, 8/7/6 → 5, 4…2 → 1.
    static func effectiveWindow(_ requested: Int) -> Int {
        guard requested >= 5 else { return 1 }
        return requested % 4 == 1 ? requested : 4 * ((requested - 1) / 4) + 1
    }

    /// Next legal length ≥ `n` for a segment-final chunk: 4k+1 for `.initializing`
    /// (1, 5, 9, …), a multiple of 4 for `.active`.
    static func paddedLength(_ n: Int, memoryState: VAEMemoryState) -> Int {
        precondition(n >= 1)
        if memoryState == .active { return 4 * ((n + 3) / 4) }
        return n == 1 ? 1 : 4 * ((n + 2) / 4) + 1
    }

    /// The effective window (4k+1, ≥ 5). T = 1 never builds a planner — the driver routes it
    /// to the pre-temporal per-frame path.
    let window: Int

    private(set) var isFirstChunkOfSegment = true
    private(set) var buffered = 0

    init(window: Int) {
        precondition(window >= 5 && window % 4 == 1,
                     "planner window must be 4k+1 and ≥ 5, got \(window)")
        self.window = window
    }

    /// Steady-state chunk length: the first chunk is `window`, every later one `window − 1`
    /// (a multiple of 4 — V12-S: at W=9 the schedule is 9, 8, 8, …).
    var targetLength: Int { isFirstChunkOfSegment ? window : window - 1 }

    /// One frame arrives. `cutBeforeThisFrame` is the detector's verdict for the transition
    /// (previous frame → this frame): true means THIS frame starts a new shot.
    mutating func ingest(cutBeforeThisFrame: Bool) -> Actions {
        var actions = Actions()
        if cutBeforeThisFrame {
            if buffered > 0 {
                actions.earlyChunk = segmentFinalChunk(frameCount: buffered)
                buffered = 0
            } else if !isFirstChunkOfSegment {
                // Boundary-aligned cut: the window closed exactly at the cut, as an interior
                // chunk. The stream still holds the old shot's tail; drop it.
                actions.resetStream = true
            }
            isFirstChunkOfSegment = true
        }
        buffered += 1
        if buffered == targetLength {
            actions.fullChunk = Chunk(frameCount: buffered, paddedCount: buffered,
                                      memoryState: isFirstChunkOfSegment ? .initializing : .active,
                                      endsSegment: false)
            buffered = 0
            isFirstChunkOfSegment = false
        }
        return actions
    }

    /// End of clip: run whatever is buffered as a segment-final chunk (nil when the last
    /// window closed exactly at the end).
    mutating func flush() -> Chunk? {
        guard buffered > 0 else { return nil }
        let chunk = segmentFinalChunk(frameCount: buffered)
        buffered = 0
        isFirstChunkOfSegment = true
        return chunk
    }

    private func segmentFinalChunk(frameCount n: Int) -> Chunk {
        let state: VAEMemoryState = isFirstChunkOfSegment ? .initializing : .active
        return Chunk(frameCount: n,
                     paddedCount: Self.paddedLength(n, memoryState: state),
                     memoryState: state,
                     endsSegment: true)
    }
}
