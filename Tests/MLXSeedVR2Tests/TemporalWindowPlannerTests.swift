// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Temporal chunking driver scheduling (GAP-PROGRAM V12-D): window arithmetic, early
// termination at cuts, ragged-tail padding, frame-count conservation. All weightless.
import CoreMedia
import XCTest

@testable import MLXSeedVR2
import SeedVR2MLX

final class TemporalWindowPlannerTests: XCTestCase {

    // MARK: - Arithmetic

    func testEffectiveWindowRoundsDownToLegal() {
        // Legal = 1 or 4k+1. Everything else rounds DOWN (never up past the user's knob).
        let cases: [(Int, Int)] = [(13, 13), (12, 9), (11, 9), (10, 9), (9, 9), (8, 5), (7, 5),
                                   (6, 5), (5, 5), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1), (-3, 1),
                                   (17, 17)]
        for (requested, expected) in cases {
            XCTAssertEqual(TemporalWindowPlanner.effectiveWindow(requested), expected,
                           "effectiveWindow(\(requested))")
        }
    }

    func testPaddedLengthPerMemoryState() {
        // Segment-final chunks pad to the next legal length: 4k+1 when `.initializing`
        // (the first chunk carries the encoder's first-frame special case), 4k when `.active`.
        let initCases: [(Int, Int)] = [(1, 1), (2, 5), (3, 5), (4, 5), (5, 5), (6, 9), (8, 9),
                                       (9, 9), (10, 13)]
        for (n, expected) in initCases {
            XCTAssertEqual(TemporalWindowPlanner.paddedLength(n, memoryState: .initializing),
                           expected, "init pad(\(n))")
        }
        let activeCases: [(Int, Int)] = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 8), (8, 8), (9, 12)]
        for (n, expected) in activeCases {
            XCTAssertEqual(TemporalWindowPlanner.paddedLength(n, memoryState: .active),
                           expected, "active pad(\(n))")
        }
    }

    func testLatentFrameArithmetic() {
        // latT = 1 + (T−1)/4 on the first chunk, T/4 after — the V12-S slicing rule.
        let first = TemporalWindowPlanner.Chunk(frameCount: 9, paddedCount: 9,
                                                memoryState: .initializing, endsSegment: false)
        XCTAssertEqual(first.latentFrames, 3)
        let steady = TemporalWindowPlanner.Chunk(frameCount: 8, paddedCount: 8,
                                                 memoryState: .active, endsSegment: false)
        XCTAssertEqual(steady.latentFrames, 2)
        let tail = TemporalWindowPlanner.Chunk(frameCount: 3, paddedCount: 4,
                                               memoryState: .active, endsSegment: true)
        XCTAssertEqual(tail.latentFrames, 1)
    }

    // MARK: - Scheduling scenarios (chunks harvested by simulation)

    private struct Run {
        var chunks: [TemporalWindowPlanner.Chunk] = []
        var resets = 0   // boundary-aligned cut resets (no chunk to run)
    }

    private func simulate(window: Int, frames: Int, cuts: Set<Int>) -> Run {
        var planner = TemporalWindowPlanner(window: window)
        var run = Run()
        for i in 0 ..< frames {
            let actions = planner.ingest(cutBeforeThisFrame: cuts.contains(i))
            if let early = actions.earlyChunk { run.chunks.append(early) }
            if actions.resetStream { run.resets += 1 }
            if let full = actions.fullChunk { run.chunks.append(full) }
        }
        if let tail = planner.flush() { run.chunks.append(tail) }
        return run
    }

    func testUncutClipSchedulesFirstThenSteadyChunks() {
        // 65 frames at W=9 → 9, 8×7 — the V12-S schedule; closes exactly, nothing padded.
        let run = simulate(window: 9, frames: 65, cuts: [])
        XCTAssertEqual(run.chunks.map(\.frameCount), [9] + Array(repeating: 8, count: 7))
        XCTAssertEqual(run.chunks.map(\.paddedCount), run.chunks.map(\.frameCount), "no pads")
        XCTAssertEqual(run.chunks[0].memoryState, .initializing)
        XCTAssertTrue(run.chunks.dropFirst().allSatisfy { $0.memoryState == .active })
        XCTAssertTrue(run.chunks.allSatisfy { !$0.endsSegment })
        XCTAssertEqual(run.resets, 0)
    }

    func testRaggedTailPadsAndEndsSegment() {
        // 20 frames at W=9 → 9, 8 interior + a 3-frame tail padded to 4 (active).
        let run = simulate(window: 9, frames: 20, cuts: [])
        XCTAssertEqual(run.chunks.map(\.frameCount), [9, 8, 3])
        XCTAssertEqual(run.chunks.last!.paddedCount, 4)
        XCTAssertEqual(run.chunks.last!.memoryState, .active)
        XCTAssertTrue(run.chunks.last!.endsSegment)
    }

    func testShortClipIsOneInitializingPaddedChunk() {
        let run = simulate(window: 9, frames: 3, cuts: [])
        XCTAssertEqual(run.chunks.map(\.frameCount), [3])
        XCTAssertEqual(run.chunks[0].paddedCount, 5)
        XCTAssertEqual(run.chunks[0].memoryState, .initializing)
        XCTAssertTrue(run.chunks[0].endsSegment)
    }

    func testBoundaryAlignedCutResetsWithoutAChunk() {
        // Cut exactly where a window closed (frame 9 at W=9): nothing is buffered, but the
        // stream still carries the old shot's tail — a reset must fire, and the new segment
        // restarts the first-chunk arithmetic (9 frames, `.initializing`).
        let run = simulate(window: 9, frames: 33, cuts: [9])
        XCTAssertEqual(run.resets, 1)
        XCTAssertEqual(run.chunks.map(\.frameCount), [9, 9, 8, 7])
        XCTAssertEqual(run.chunks.map(\.memoryState),
                       [.initializing, .initializing, .active, .active])
        XCTAssertEqual(run.chunks.map(\.endsSegment), [false, false, false, true])
        XCTAssertEqual(run.chunks.last!.paddedCount, 8)
    }

    func testMidWindowCutTerminatesEarly() {
        // The N11 load-bearing case: cut at frame 13 inside the second window (W=9). The 4
        // buffered frames [9,13) run as a segment-final chunk BEFORE frame 13 joins — a flush
        // at the next join would leave them jointly processed with the old shot.
        let run = simulate(window: 9, frames: 33, cuts: [13])
        XCTAssertEqual(run.chunks.map(\.frameCount), [9, 4, 9, 8, 3])
        let early = run.chunks[1]
        XCTAssertTrue(early.endsSegment)
        XCTAssertEqual(early.memoryState, .active)
        XCTAssertEqual(early.paddedCount, 4)
        // The cut frame starts a fresh segment: first-chunk arithmetic again.
        XCTAssertEqual(run.chunks[2].memoryState, .initializing)
        XCTAssertEqual(run.chunks[2].frameCount, 9)
        XCTAssertEqual(run.resets, 0)
    }

    func testMidFirstWindowCutPadsInitializing() {
        // Cut before the first window ever fills: the early chunk is `.initializing` and pads
        // on the 4k+1 rail (3 → 5).
        let run = simulate(window: 9, frames: 12, cuts: [3])
        XCTAssertEqual(run.chunks.map(\.frameCount), [3, 9])
        XCTAssertEqual(run.chunks[0].memoryState, .initializing)
        XCTAssertEqual(run.chunks[0].paddedCount, 5)
        XCTAssertTrue(run.chunks[0].endsSegment)
        XCTAssertEqual(run.chunks[1].memoryState, .initializing)
    }

    func testCutEveryFrameCollapsesToPerFrame() {
        // Degenerate (or detector-gone-wild) case: the driver must not break — every frame
        // becomes its own `.initializing` chunk of padded length 1, i.e. T=1-shaped behaviour.
        let run = simulate(window: 9, frames: 10, cuts: Set(0 ..< 10))
        XCTAssertEqual(run.chunks.count, 10)
        XCTAssertTrue(run.chunks.allSatisfy {
            $0.frameCount == 1 && $0.paddedCount == 1
                && $0.memoryState == .initializing && $0.endsSegment
        })
        XCTAssertEqual(run.resets, 0)
    }

    func testFrameConservationUnderArbitraryCutPatterns() {
        // Property: Σ frameCount == frames in, for every window and any cut pattern; padding
        // appears only on segment-final chunks; `.active` never follows a segment end.
        var lcg: UInt64 = 0x2545F4914F6CDD1D
        func nextBool(_ p: UInt64) -> Bool { lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            return (lcg >> 33) % 100 < p }
        for window in [5, 9, 13] {
            for density in [0 as UInt64, 3, 20, 60] {
                for frames in [1, 4, 9, 17, 33, 64] {
                    var cuts = Set<Int>()
                    for i in 0 ..< frames where nextBool(density) { cuts.insert(i) }
                    let run = simulate(window: window, frames: frames, cuts: cuts)
                    XCTAssertEqual(run.chunks.map(\.frameCount).reduce(0, +), frames,
                                   "W=\(window) frames=\(frames) cuts=\(cuts.sorted())")
                    var expectInit = true
                    for chunk in run.chunks {
                        if chunk.paddedCount != chunk.frameCount {
                            XCTAssertTrue(chunk.endsSegment, "padding only on segment-final chunks")
                        }
                        if expectInit { XCTAssertEqual(chunk.memoryState, .initializing) }
                        expectInit = chunk.endsSegment
                        // A boundary-aligned-cut reset also restarts the segment; simulate()
                        // cannot see it here, so only assert the initializing side.
                        if chunk.memoryState == .initializing, !expectInit {
                            // (nothing further — interior init chunk is legal mid-run only
                            //  after a reset, covered by the boundary-aligned test)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Budget clamp (the ladder dial)

    func testEffectiveTemporalWindowBudgetClamp() {
        // Single 256² tile: the V12-S ladder directly. 12.5 GiB fits T=9 (11.78) not T=13;
        // 10 GiB fits T=5 (9.83); 8 GiB fits only T=1.
        func t(_ requested: Int, _ gib: Double?, w: Int = 256, h: Int = 256) -> Int {
            SeedVR2UpscalePackage.effectiveTemporalWindow(
                requested: requested,
                budgetBytes: gib.map { UInt64($0 * 1_073_741_824) },
                outputWidth: w, outputHeight: h, tileSize: 256, tileOverlap: 32)
        }
        XCTAssertEqual(t(13, nil), 13, "no budget stamped → the knob")
        XCTAssertEqual(t(9, 12.5), 9)
        XCTAssertEqual(t(13, 12.5), 9, "13.47 GiB does not fit 12.5")
        XCTAssertEqual(t(9, 10.0), 5)
        XCTAssertEqual(t(9, 8.0), 1)
        XCTAssertEqual(t(1, 64.0), 1, "budget never raises the knob")
        // 512² output = 9 tile positions (clamped origins {0, 224, 256} per axis); each extra
        // tile charges ~0.82 GiB of streaming bank, so T=9 needs ~11.78 + 8·0.82 ≈ 18.3 GiB
        // and T=5 ~16.4 GiB — a 13 GiB budget collapses to T=1, 17 GiB carries T=5.
        XCTAssertEqual(t(9, 13.0, w: 512, h: 512), 1)
        XCTAssertEqual(t(9, 17.0, w: 512, h: 512), 5)
        XCTAssertEqual(t(9, 64.0, w: 512, h: 512), 9)
    }
}
