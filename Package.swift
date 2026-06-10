// swift-tools-version: 6.2
import PackageDescription

// mlx-seedvr2-swift — the MLXEngine `videoUpscale` package over SeedVR2-3B (one-step diffusion
// super-resolution). The first Video→Video transform of the visual optimization tier: decode →
// per-frame Lanczos pre-upscale + SeedVR2 tile-refinement (+ LAB color transfer) → HEVC encode.
// Thin conformance layer over the seedvr2-mlx-swift core (e2e GPU/int8 validated via Forge);
// the tile/feathered-seam machinery is shared from realesrgan-mlx-swift. Module is `MLXSeedVR2`.
let package = Package(
    name: "mlx-seedvr2-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXSeedVR2", targets: ["MLXSeedVR2"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/seedvr2-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/realesrgan-mlx-swift.git", from: "0.2.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
    ],
    targets: [
        .target(
            name: "MLXSeedVR2",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "SeedVR2MLX", package: "seedvr2-mlx-swift"),
                .product(name: "RealESRGANMLX", package: "realesrgan-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            // The cores (MLX) aren't Sendable-audited; the engine serializes lifecycle on
            // InferenceActor, so v5 mode keeps region-isolation a warning — same as siblings.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXSeedVR2Tests",
            dependencies: [
                "MLXSeedVR2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
