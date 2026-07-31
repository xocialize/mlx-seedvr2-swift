// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_latent_creator.py.
import MLX
import MLXRandom

public enum SeedVR2LatentCreator {
    /// Seeded Gaussian noise latents. MLX-Swift and MLX-Python share the same RNG core,
    /// so `key(seed)` produces identical noise. `latentFrames` is the latent T
    /// (1 + (T-1)/4 for a T-frame input); the default 1 is the single-frame path.
    public static func noiseLatents(seed: UInt64, height: Int, width: Int,
                                    batch: Int = 1, latentChannels: Int = 16,
                                    latentFrames: Int = 1) -> MLXArray {
        MLXRandom.normal([batch, latentChannels, latentFrames, height, width], key: MLXRandom.key(seed))
    }

    /// Append an all-ones mask channel to the encoded latent: [B,16,T,H,W] -> [B,17,T,H,W].
    /// The mask spans ALL T latent frames (SeedVR `get_condition`, task="sr": every frame is
    /// LQ-conditioned restoration — unlike i2v/v2v, which mask only the leading frames).
    public static func condition(_ encoded: MLXArray) -> MLXArray {
        let l = encoded.ndim == 4 ? encoded.expandedDimensions(axis: 2) : encoded
        let (t, h, w) = (l.shape[2], l.shape[3], l.shape[4])
        let mask = MLXArray.ones([1, 1, t, h, w]).asType(l.dtype)
        return concatenated([l, mask], axis: 1)
    }
}
