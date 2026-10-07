# mlx-seedvr2-swift

The MLXEngine **`videoUpscale`** package over [SeedVR2-3B](https://github.com/xocialize/seedvr2-mlx-swift) (ByteDance, one-step diffusion super-resolution) — the first **Video → Video** transform of the visual optimization tier.

Per frame: CoreImage **Lanczos pre-upscale** (the spatial 2×/4×) → SeedVR2 one-step diffusion
**refinement** at 1:1, tile-blended with feathered seams (shared `MLXTileProcessor`) →
**LAB-wavelet color transfer** toward the upscaled base (mflux parity). Frames stream
decode → refine → encode (**HEVC, BT.709-tagged**) so memory stays bounded; cancellation is
honored per frame.

**Temporal by default (v0.8.0, `temporalWindow`, default 9).** The video surface schedules
decoded frames into T-frame windows and refines each window through the T>1 causal VAE with
**streaming memory across the joins** (V12-S — the chunk-seam fix), tiled spatially with one
streaming bank per tile position. A **scene cut** (media-bridge `SceneCutDetector`, inline on
decoded frames) **ends the current window early and flushes the stream** — flushing only at the
next join measurably cannot repair the straddling frames (GAP-PROGRAM N11). `temporalWindow: 1`
is the pre-temporal per-frame path above, bit-identical to v0.7.x. Memory ladder (one 256² tile
stream, MLX cache capped): T = 1/5/9/13 → 7.82 / 9.83 / 11.78 / 13.47 GiB `phys_footprint`;
each additional tile position holds ~784 MiB of streaming bank, and a stamped
`availableBudgetBytes` clamps T down the ladder automatically.

## Weights

| Repo | Quant | Size (4 files) | Default |
|---|---|---|---|
| `mlx-community/SeedVR2-3B-mlx-int8` | int8 | 4.72 GB | ✅ (near-lossless, ~50.3 dB; e2e GPU-validated via Forge) |
| `mlx-community/SeedVR2-3B-mlx` | fp16 | 8.44 GB | |

int4 is deliberately not offered (degrades to ~22.7 dB). Each repo carries the same four weight files
— `config.json`, `pos_emb.safetensors`, `vae.safetensors`, `transformer.safetensors` — and nothing else
is fetched.

### Who downloads, and where (v0.10.0)

`SeedVR2Configuration` conforms to `ModelStorable` + `WeightSourcing` (engine ≥ 0.32.0, contract 1.24):
it declares **one source per quant lane** — role `seedvr2-int8` or `seedvr2-fp16`, matching only that
repo's four files — and with a model store attached, **the engine** downloads it before `load()`, into
the store's flat layout `<store>/models--mlx-community--SeedVR2-3B-mlx-int8/`, surfacing its
`.downloading` phase. `load()` only resolves the directory, in this order:

1. **`snapshotDirectory`** (explicit) — read as-is; never touches the network. The dev-mode escape hatch.
2. **No store attached** — the core's own cache, `~/Library/Caches/seedvr2-mlx/<org>--<name>/`,
   downloaded by `load()` itself if absent. Unchanged from v0.9.x (and, as before, without progress).
3. **Store attached** — the flat layout, else a hub-client snapshot (`refs/main` → `snapshots/<commit>/`),
   else a **v0.9.x in-store snapshot** (below). If none is complete — only when the package is driven
   without the engine's materialization pass — `load()` downloads into the flat layout itself.

The declared source follows `BudgetAware`: an fp16 configuration stamped with < ~11 GB of headroom
declares, downloads and loads the int8 repo. `repoOverride`, when set, is what is declared and loaded.

### Existing snapshots: honored, then adopted — never re-downloaded

v0.9.x downloaded with the core's own `HFHub.snapshot`, into `<store>/seedvr2-mlx/<org>--<name>/`
whenever the engine stamped a store root (e.g. Forge), else into `~/Library/Caches/seedvr2-mlx/`.
**Decision:** every copy v0.9.x would have reused is still reused.

- **`<store>/seedvr2-mlx/<org>--<name>/` with all four files** satisfies the source, so the engine does
  not download, and `load()` **adopts** it: the four files are *moved* into
  `<store>/models--<org>--<name>/` (beside the install marker the engine already writes there) and the
  emptied `seedvr2-mlx` directories are removed. Same store root ⇒ same volume ⇒ each move is a
  rename: no bytes copied. Why adopt rather than read in place: in place, the storage panel credits the
  repo's marker with an empty directory, and removing the model from the store would leave ~4.7 GB
  behind. If the volumes differ or any move fails, the moves already made are put back and the
  snapshot is loaded in place — never a copy, never a download.
- **`~/Library/Caches/seedvr2-mlx/`** is used exactly as before: only when **no** store is attached.
  With a store attached it is not a candidate — v0.9.x never read it in that case either (it downloaded
  into the store), and counting it would leave the weights outside the store the user picked.

⚠️ After adoption, an app still on v0.9.x pointed at the *same* store no longer finds
`seedvr2-mlx/…` and downloads into it again. Upgrade every consumer of a shared store together.

The offline **MAT-1..5** gate runs per quant lane in `Tests/MLXSeedVR2Tests/MaterializationTests.swift`,
beside the layout and adoption tests (all weightless). The live lane is the smoke CLI through the real
engine: `seedvr2-package-smoke --engine-store <dir> --expect-download yes|no --image in.png --out out.png`
prints `MLXEngineTestKit`'s `[MAT]` line and the resulting store layout.

## Usage

```swift
import MLXServeCore
import MLXSeedVR2

let engine = MLXServeEngine()
await engine.useModelStore(ModelStore(root: modelsFolder))   // the engine downloads into it on first prepare
try await engine.register(SeedVR2UpscalePackage.registration, configuration: SeedVR2Configuration())

let resp = try await engine.run(VideoUpscaleRequest(video: clip, scale: 2)) as! VideoUpscaleResponse
// resp.video — 2× HEVC .mp4; resp.appliedScale == 2
```

Requirements: macOS 26+ (Apple Silicon, Metal GPU; Pro-tier chip floor — 3B-param diffusion per
tile per frame). Port MIT; weights Apache-2.0 (ByteDance-Seed).
