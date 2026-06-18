// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
import ArgumentParser
import Foundation
import SeedVR2MLX

struct SeedVR2Upscale: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "seedvr2-upscale",
        abstract: "Upscale an image with SeedVR2 (MLX-Swift). [WIP — modules landing]"
    )

    @Option(name: .long, help: "Directory with exported weights (transformer/vae/pos_emb/config).")
    var weights: String

    @Option(name: .shortAndLong, help: "Input image path.")
    var image: String = ""

    func run() throws {
        let w = try SeedVR2Weights(directory: URL(fileURLWithPath: weights))
        print("Loaded SeedVR2 weights: \(w.transformer.count) transformer tensors, "
            + "\(w.vae.count) vae tensors, vidDim=\(w.config.vidDim), layers=\(w.config.numLayers)")
        print("Pipeline assembly is in progress — see docs/PORT-PLAN.md for module status.")
    }
}

SeedVR2Upscale.main()
