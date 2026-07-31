// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// The MEMORY MODEL of streaming memory, pinned as a test rather than left as a probe receipt.
//
// V12-D measured the streaming banks at "~0.82 GiB per spatial tile position" and that phrasing —
// accurate for the geometry it was measured at — invites two wrong inferences that this file exists
// to close off. Both were reached for during the V13-T follow-up before the shapes were checked:
//
//   ❌ "Use fewer, bigger tiles."   The tail keeps the tile's FULL spatial extent, so a 2× tile side
//                                   holds a 4× tail. Positions × area is just area. Leading order,
//                                   nothing is saved.
//   ❌ "Lower the temporal window."  `memoryFrames = kernel - stride` is fixed by the architecture,
//                                   so tails are IDENTICAL in T. (V12-S measured this; here it is
//                                   pinned rather than remembered.)
//
// 🔑 The honest unit is **bytes per output pixel**, and it is a property of the architecture. These
// tests need no checkpoint — tail SHAPES are independent of weight values — so the model is pinned
// weightlessly and in CI, which is exactly what a load-bearing sizing claim should cost.
//
// ⚠️ If one of these fails after a VAE change, the sizing guidance in ForgeCore's REQUIREMENTS.md and
// `EngineVideoUpscaler.videoUpscaleFits` is stale — those are downstream of this constant.
import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import SeedVR2MLX

final class VAEStreamingBankSizeTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: Device(.cpu)) }

    /// Randomized parameters: tail shapes do not depend on weight values, but zeros would make any
    /// value-sensitive follow-up vacuous, and this matches the sibling suite's construction.
    ///
    /// 🚨 **The `.asType(VAEPrecision.dtype)` on BOTH the parameters and the input is load-bearing,
    /// and it took two attempts to get right.** `MLXRandom.uniform` yields float32, and — the part
    /// that is genuinely easy to miss — a freshly constructed `SeedVR2VAE()` holds **float32**
    /// parameters: bfloat16 arrives only when the real loader casts them. So `asType(value.dtype)`,
    /// which looks like the careful thing to write, is a no-op here and leaves the whole probe running
    /// fp32. The tail is recorded from the conv's INPUT (before its own cast to `weight.dtype`), so
    /// an fp32 input yields fp32 tails and the probe reports **twice** the bytes the shipping path
    /// retains.
    ///
    /// 🔑 The dtype assertion in `bankBytes` is what caught this, and is the reason it is an assertion
    /// rather than a comment. A sizing probe that quietly runs at the wrong precision is worse than no
    /// probe — it produces a clean, plausible number for a configuration nobody ships. (The basis
    /// error `ForgeCore/REQUIREMENTS.md` already warns about twice: state the measurement
    /// CONFIGURATION, not just the units.)
    private func makeRandomizedVAE() throws -> SeedVR2VAE {
        let vae = SeedVR2VAE()
        var params: [(String, MLXArray)] = []
        for (i, (key, value)) in vae.parameters().flattened().enumerated() {
            params.append((key, MLXRandom.uniform(low: -0.05, high: 0.05, value.shape,
                                                  key: MLXRandom.key(UInt64(2000 + i)))
                                    .asType(VAEPrecision.dtype)))
        }
        try vae.update(parameters: ModuleParameters.unflattened(params), verify: .none)
        return vae
    }

    /// Total retained-tail bytes after streaming one chunk of `frames` at `side`×`side`.
    private func bankBytes(_ vae: SeedVR2VAE, side: Int, frames: Int) -> Int {
        vae.resetStreamingMemory()
        let x = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 3, frames, side, side],
                                  key: MLXRandom.key(11)).asType(VAEPrecision.dtype)
        let z = vae.encode(x, memoryState: .initializing)
        _ = vae.decode(z, memoryState: .initializing)
        let tails = vae.streamingMemoryTails()
        eval(tails)
        // Guard the precision this probe depends on: a tail wider than the VAE's own compute dtype
        // means the measurement no longer describes the shipping configuration.
        for t in tails {
            XCTAssertEqual(t.dtype, VAEPrecision.dtype,
                           "retained tail is \(t.dtype), not the VAE's \(VAEPrecision.dtype) — "
                           + "this probe would report the wrong size for the shipping path")
        }
        let hist = Dictionary(grouping: tails, by: { "\($0.dtype)" })
            .mapValues { ($0.count, $0.reduce(0) { $0 + $1.size }) }
        print("  side=\(side) T=\(frames): \(tails.count) tails, "
              + "\(hist.map { "\($0.key) x\($0.value.0) = \($0.value.1) elems" }.sorted().joined(separator: ", "))")
        return tails.reduce(0) { $0 + $1.nbytes }
    }

    /// 🔑 **Tail bytes scale with tile AREA — so a coarser tile grid saves nothing.**
    ///
    /// Measured at three sides — **57 retained tails, 6,429,952 elements, all bfloat16** — the state
    /// fits **`12525·s² + 1024·s + 1536` bytes** exactly.
    ///
    /// ⚠️ It is *not* purely quadratic, and the residual is informative rather than noise: some
    /// layer's spatial extent carries a `+1` from padding, and `(s/k + 1)²` expands to exactly a
    /// quadratic-plus-linear-plus-constant. The non-quadratic part is **0.26% at s=32 and 0.03% at
    /// s=256** — it shrinks as the tile grows, so the sizing rule is quadratic at every tile size
    /// anyone would ship.
    ///
    /// 🔑 **This is the independent confirmation of V12-D's process-level number.** The fit at a 256²
    /// tile gives **0.765 GiB**; V12-D measured **~0.82 GiB** per tile position from `phys_footprint`
    /// deltas on the shipping path. Two unrelated methods — array shapes here, process footprint
    /// there — agreeing within 7% is what makes the per-output-pixel model safe to size against, and
    /// the direction of the gap is the expected one (the process figure carries per-position overhead
    /// beyond the tails themselves).
    ///
    /// 🚨 **An earlier version of this test reported `14394·s² + …` and 0.879 GiB. That was a
    /// MIXED-PRECISION configuration, not fp32 and not the shipping path** — with fp32 parameters the
    /// convs fed by a `norm` still cast to bfloat16, so 974,080 elements stayed fp32 and 5,455,872 did
    /// not. Element count was identical in both runs; only the widths differed. Worth stating because
    /// the wrong number looked *more* conservative and would have passed unchallenged.
    func testTailBytesScaleWithTileArea() throws {
        let vae = try makeRandomizedVAE()
        let b32 = bankBytes(vae, side: 32, frames: 5)
        let b64 = bankBytes(vae, side: 64, frames: 5)
        let b128 = bankBytes(vae, side: 128, frames: 5)

        XCTAssertGreaterThan(b32, 0, "no tails were retained — the probe is measuring nothing")

        // Quadratic to within the padding term, which is <0.5% at the smallest side tested.
        XCTAssertEqual(Double(b64) / Double(b32), 4.0, accuracy: 0.02,
                       "tail bytes must scale with tile AREA (32→64 is 4× the pixels)")
        XCTAssertEqual(Double(b128) / Double(b32), 16.0, accuracy: 0.08,
                       "tail bytes must scale with tile AREA (32→128 is 16×)")

        // Pin the closed form, so a VAE change that alters the sizing model fails here loudly
        // rather than silently invalidating ForgeCore's admission gate.
        func predicted(_ s: Int) -> Int { 12525 * s * s + 1024 * s + 1536 }
        XCTAssertEqual(b32, predicted(32), "streaming-bank size law changed")
        XCTAssertEqual(b64, predicted(64), "streaming-bank size law changed")
        XCTAssertEqual(b128, predicted(128), "streaming-bank size law changed")

        // 0.765 GiB per 256² tile position — the number ForgeCore sizes against, derived rather
        // than assumed, and within 7% of V12-D's independent process-footprint measurement.
        let at256 = Double(predicted(256)) / 1_073_741_824.0
        XCTAssertEqual(at256, 0.765, accuracy: 0.01)
    }

    /// 🔑 **Tail bytes are INVARIANT in the temporal window — so clamping T does not shrink them.**
    ///
    /// `memoryFrames = kernel.0 - stride.0` per conv, fixed by the architecture. This is what makes
    /// the "784 MiB, constant in T" row in the V12-S receipt a structural fact rather than an
    /// observation that happened to hold over the T values tried.
    func testTailBytesAreInvariantInTemporalWindow() throws {
        let vae = try makeRandomizedVAE()
        let b5 = bankBytes(vae, side: 32, frames: 5)
        let b9 = bankBytes(vae, side: 32, frames: 9)
        let b13 = bankBytes(vae, side: 32, frames: 13)
        XCTAssertEqual(b5, b9, "retained tails must not grow with T")
        XCTAssertEqual(b5, b13, "retained tails must not grow with T")
    }

    /// The one geometry lever that IS real, stated so it is not confused with the two that are not:
    /// overlap redundancy. Banks cost `total tile area`, not `output area`, so a tiling that covers
    /// the frame with less overlap costs proportionally less — `(tile / (tile - overlap))²`.
    ///
    /// ⚠️ This is a **second-order** effect and must not be sold as a fix: at the shipping 256/32 the
    /// redundancy is only 1.31×, so even a perfect tiling recovers ~23% of bank bytes, against a
    /// requirement that is multiples over budget on real content. Bank offload/eviction is the
    /// first-order lever. This test pins the arithmetic so the two are not conflated.
    func testOverlapRedundancyIsTheOnlyGeometryLeverAndIsSecondOrder() {
        func redundancy(tile: Int, overlap: Int) -> Double {
            let step = Double(tile - overlap)
            return (Double(tile) / step) * (Double(tile) / step)
        }
        let shipping = redundancy(tile: 256, overlap: 32)     // 1.306…
        let coarser  = redundancy(tile: 512, overlap: 32)     // 1.138…
        XCTAssertEqual(shipping, 1.3061, accuracy: 0.001)
        XCTAssertEqual(coarser, 1.1384, accuracy: 0.001)

        // Going 256→512 recovers 1 - 1.1384/1.3061 ≈ 12.8% of bank bytes. Real, and nowhere near
        // the several-fold reduction a 16 GB machine needs on real content.
        let recovered = 1.0 - coarser / shipping
        XCTAssertLessThan(recovered, 0.15,
                          "if enlarging the tile ever recovers >15%, revisit — but it cannot, "
                          + "because redundancy is bounded below by 1.0 and 256/32 is only 1.31")
    }
}
