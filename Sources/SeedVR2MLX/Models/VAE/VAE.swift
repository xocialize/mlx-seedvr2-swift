// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_vae/{encoder,decoder,vae}.py.
import MLX
import MLXNN

public final class Encoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock3D]
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(inChannels: Int = 3, outChannels: Int = 16,
                blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, temporalDownBlocks: Int = 2) {
        self._convIn.wrappedValue = CausalConv3d(inChannels, blockOut[0])
        var blocks: [DownBlock3D] = []
        var outCh = blockOut[0]
        let n = blockOut.count
        for (i, ch) in blockOut.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalDown = (i >= n - temporalDownBlocks - 1) && !isFinal
            blocks.append(DownBlock3D(inCh, outCh, numLayers: layersPerBlock, addDownsample: !isFinal, temporalDown: temporalDown))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = MidBlock3D(blockOut[n - 1])
        self._convNormOut.wrappedValue = groupNorm32(blockOut[n - 1])
        self._convOut.wrappedValue = CausalConv3d(blockOut[n - 1], 2 * outChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, memoryState: VAEMemoryState = .disabled) -> MLXArray {
        var h = convIn(x, memoryState: memoryState)
        for b in downBlocks { h = b(h, memoryState: memoryState) }
        h = midBlock(h, memoryState: memoryState)
        h = silu(vaeGroupNorm(h, convNormOut))
        return convOut(h, memoryState: memoryState)
    }

    var causalConvs: [CausalConv3d] {
        [convIn] + downBlocks.flatMap(\.causalConvs) + midBlock.causalConvs + [convOut]
    }
}

public final class Decoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock3D
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock3D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(inChannels: Int = 16, outChannels: Int = 3,
                blockOut: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 3, temporalUpBlocks: Int = 2) {
        let rev = Array(blockOut.reversed())
        self._convIn.wrappedValue = CausalConv3d(inChannels, rev[0])
        self._midBlock.wrappedValue = MidBlock3D(rev[0])
        var blocks: [UpBlock3D] = []
        var outCh = rev[0]
        let n = rev.count
        for (i, ch) in rev.enumerated() {
            let inCh = outCh; outCh = ch
            let isFinal = i == n - 1
            let temporalUp = i < temporalUpBlocks
            blocks.append(UpBlock3D(inCh, outCh, numLayers: layersPerBlock, addUpsample: !isFinal, temporalUp: temporalUp))
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = groupNorm32(rev[n - 1])
        self._convOut.wrappedValue = CausalConv3d(rev[n - 1], outChannels)
        super.init()
    }

    public func callAsFunction(_ z: MLXArray, memoryState: VAEMemoryState = .disabled) -> MLXArray {
        var h = convIn(z, memoryState: memoryState)
        h = midBlock(h, memoryState: memoryState)
        for b in upBlocks { h = b(h, memoryState: memoryState) }
        h = silu(vaeGroupNorm(h, convNormOut))
        return convOut(h, memoryState: memoryState)
    }

    var causalConvs: [CausalConv3d] {
        [convIn] + midBlock.causalConvs + upBlocks.flatMap(\.causalConvs) + [convOut]
    }
}

public final class SeedVR2VAE: Module {
    @ModuleInfo(key: "encoder") var encoder: Encoder3D
    @ModuleInfo(key: "decoder") var decoder: Decoder3D
    let scalingFactor: Float = 0.9152
    let latentChannels = 16

    public override init() {
        self._encoder.wrappedValue = Encoder3D()
        self._decoder.wrappedValue = Decoder3D()
        super.init()
    }

    /// x [B,3,T,H,W] (or [B,3,H,W]) -> latent [B,16,T,H/8,W/8].
    ///
    /// `memoryState` drives SeedVR's streaming mode (`slicing_encode`): `.initializing` for the
    /// first chunk of a clip (T ≡ 1 mod 4 — it carries the causal first-frame special case),
    /// `.active` for every chunk after (T ≡ 0 mod 4, latT = T/4). `.disabled` — the default — is
    /// the single-pass path and is bit-identical to the pre-streaming code.
    public func encode(_ xIn: MLXArray, memoryState: VAEMemoryState = .disabled) -> MLXArray {
        let x = xIn.ndim == 4 ? xIn.expandedDimensions(axis: 2) : xIn
        let h = encoder(x, memoryState: memoryState)
        let mean = h[0..., 0 ..< latentChannels]
        let out = mean * scalingFactor
        settle(out, encoder.causalConvs, memoryState)
        return out
    }

    /// z [B,16,T,H,W] -> image [B,3,T,H*8,W*8].
    ///
    /// Frame count follows the memory state, exactly as `slicing_decode` does: `.disabled` /
    /// `.initializing` give the causal inverse 4·(latT−1)+1, `.active` gives 4·latT (the
    /// first-frame duplicate is only dropped on the chunk that has one).
    public func decode(_ zIn: MLXArray, memoryState: VAEMemoryState = .disabled) -> MLXArray {
        var z = zIn.ndim == 4 ? zIn.expandedDimensions(axis: 2) : zIn
        z = z / scalingFactor
        let out = decoder(z, memoryState: memoryState)
        settle(out, decoder.causalConvs, memoryState)
        return out
    }

    /// Materialise a stage's output together with the streaming tails it just produced.
    ///
    /// 🚨 Scope, not tidiness. A tail is a lazy slice of a conv input, so an unevaluated tail pins
    /// that entire activation. Evaluating the ENCODER's tails only at the end of `upscale` held
    /// every encoder intermediate alive across the diffusion step and the whole decode: measured
    /// at 256²/T=5, peak `phys_footprint` 21.24 -> 20.13 GiB (uncapped MLX buffer cache) when
    /// each stage settles its own instead. Doing it here costs nothing — the tails share the
    /// output's graph, so this is the eval that was going to happen anyway, at the right moment.
    ///
    /// ⚠️ Neither this nor `.contiguous()` is what makes the streaming path affordable: most of
    /// that 20 GiB is MLX buffer cache, not working set. Bounded at 2 GiB (what the shipping
    /// engine does) the same run is 9.83 GiB, of which the tails are 784 MiB. See V12-S §5.
    private func settle(_ out: MLXArray, _ convs: [CausalConv3d], _ memoryState: VAEMemoryState) {
        guard memoryState != .disabled else { return }
        eval([out] + convs.compactMap(\.memory.tail))
    }

    /// Every causal conv that carries streaming state, encoder then decoder.
    var causalConvs: [CausalConv3d] { encoder.causalConvs + decoder.causalConvs }

    /// Drop all streaming tails — call between clips (or when abandoning one part-way).
    public func resetStreamingMemory() {
        for c in causalConvs { c.memory.tail = nil }
    }

    /// The streaming tails as arrays, for the caller to `eval` alongside the chunk output.
    ///
    /// Each tail is a lazy slice of a conv input, so leaving it unevaluated pins that whole
    /// intermediate — up to a full-resolution decoder activation — alive until the next chunk.
    /// Evaluating them with the output costs nothing extra (they share its graph) and drops
    /// everything but the few frames actually retained.
    public func streamingMemoryTails() -> [MLXArray] {
        causalConvs.compactMap(\.memory.tail)
    }

    /// Snapshot the current streaming state (every causal conv's tail, in `causalConvs` order).
    ///
    /// **Why this exists: one VAE, several interleaved temporal streams.** A tiled temporal
    /// driver refines the same clip tile position by tile position, and each spatial tile is its
    /// own causal stream — carrying tile A's tails into tile B's chunk would condition B on A's
    /// pixels. The driver exports the bank after each tile's chunk and adopts that tile's bank
    /// before its next one. Reference-copy cheap; the arrays themselves are the settled tails.
    public func exportStreamingMemory() -> VAEStreamingBank {
        VAEStreamingBank(tails: causalConvs.map(\.memory.tail))
    }

    /// Install a previously exported streaming state (`nil` clears — same as
    /// ``resetStreamingMemory()``). The bank must come from a VAE with the same architecture;
    /// tails are re-installed positionally onto `causalConvs`.
    public func adoptStreamingMemory(_ bank: VAEStreamingBank?) {
        guard let bank else { return resetStreamingMemory() }
        precondition(bank.tails.count == causalConvs.count,
                     "streaming bank holds \(bank.tails.count) tails; VAE has \(causalConvs.count) causal convs")
        for (conv, tail) in zip(causalConvs, bank.tails) { conv.memory.tail = tail }
    }
}

/// Opaque snapshot of a `SeedVR2VAE`'s streaming memory (see
/// ``SeedVR2VAE/exportStreamingMemory()``). One bank per independent temporal stream — e.g. one
/// per spatial tile position in a tiled temporal driver.
public final class VAEStreamingBank {
    let tails: [MLXArray?]
    init(tails: [MLXArray?]) { self.tails = tails }
}
