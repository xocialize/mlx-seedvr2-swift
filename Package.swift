// swift-tools-version: 6.2
import PackageDescription

// mlx-seedvr2-swift — SeedVR2-3B one-step diffusion super-resolution for MLXEngine.
// ONE repo, multiple products:
//   • SeedVR2MLX     — engine-agnostic Swift/MLX core (no MLXToolKit dep; usable standalone)
//   • seedvr2-upscale — the core's standalone CLI
//   • MLXSeedVR2     — the MLXEngine ModelPackage: ONE SeedVR2UpscalePackage exposes BOTH
//     imageUpscale (Export/diffusion tier) AND videoUpscale surfaces from one loaded 3B core.
//   • seedvr2-package-smoke — drives the package through the engine's load()/run() seam (gate).
// Consolidated 2026-06-18: the former standalone `seedvr2-mlx-swift` core was folded in (archived).
// The tile/feathered-seam machinery is shared from the consolidated `mlx-realesrgan-swift` (RealESRGANMLX).
let package = Package(
    name: "mlx-seedvr2-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "SeedVR2MLX", targets: ["SeedVR2MLX"]),
        .executable(name: "seedvr2-upscale", targets: ["RunUpscale"]),
        .library(name: "MLXSeedVR2", targets: ["MLXSeedVR2"]),
        .executable(name: "seedvr2-package-smoke", targets: ["SeedVR2PackageSmoke"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.17.0"),
        .package(url: "https://github.com/xocialize/frame-stream-native.git", from: "0.3.0"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
        // RealESRGANMLX now ships from the consolidated mlx-realesrgan-swift (was realesrgan-mlx-swift, archived).
        .package(url: "https://github.com/xocialize/mlx-realesrgan-swift.git", from: "0.2.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.2" ..< "0.32.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        // Engine-agnostic core (folded in from seedvr2-mlx-swift) — NO MLXToolKit dep.
        .target(
            name: "SeedVR2MLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/SeedVR2MLX",
            // Core was authored in Swift 5 (tools 5.9); keep v5 mode in this 6.2 package.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "RunUpscale",
            dependencies: [
                "SeedVR2MLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/RunUpscale",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // MLXEngine `videoUpscale` wrapper over the local core (+ shared tiling from RealESRGANMLX).
        .target(
            name: "MLXSeedVR2",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FrameStreamNative", package: "frame-stream-native"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
                "SeedVR2MLX",
                .product(name: "RealESRGANMLX", package: "mlx-realesrgan-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            // The cores (MLX) aren't Sendable-audited; the engine serializes lifecycle on
            // InferenceActor, so v5 mode keeps region-isolation a warning — same as siblings.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "SeedVR2PackageSmoke",
            dependencies: [
                "MLXSeedVR2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "SeedVR2MLXTests",
            dependencies: [
                "SeedVR2MLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/SeedVR2MLXTests"
        ),
        .testTarget(
            name: "MLXSeedVR2Tests",
            dependencies: [
                "MLXSeedVR2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FrameStreamNative", package: "frame-stream-native"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
