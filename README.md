# mlx-seedvr2-swift

The MLXEngine **`videoUpscale`** package over [SeedVR2-3B](https://github.com/xocialize/seedvr2-mlx-swift) (ByteDance, one-step diffusion super-resolution) — the first **Video → Video** transform of the visual optimization tier.

Per frame: CoreImage **Lanczos pre-upscale** (the spatial 2×/4×) → SeedVR2 one-step diffusion
**refinement** at 1:1, tile-blended with feathered seams (shared `MLXTileProcessor`) →
**LAB-wavelet color transfer** toward the upscaled base (mflux parity). Frames stream
decode → refine → encode (**HEVC, BT.709-tagged**) so memory stays bounded; cancellation is
honored per frame.

## Weights

| Repo | Quant | Default |
|---|---|---|
| `mlx-community/SeedVR2-3B-mlx-int8` | int8 | ✅ (e2e GPU-validated via Forge) |
| `mlx-community/SeedVR2-3B-mlx` | bf16 | |

## Usage

```swift
import MLXServeCore
import MLXSeedVR2

let engine = MLXServeEngine()
try await engine.register(SeedVR2VideoUpscalePackage.registration, configuration: SeedVR2Configuration())

let resp = try await engine.run(VideoUpscaleRequest(video: clip, scale: 2)) as! VideoUpscaleResponse
// resp.video — 2× HEVC .mp4; resp.appliedScale == 2
```

Requirements: macOS 26+ (Apple Silicon, Metal GPU; Pro-tier chip floor — 3B-param diffusion per
tile per frame). Port MIT; weights Apache-2.0 (ByteDance-Seed).
