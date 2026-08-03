// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
//
// Persistence for `VAEStreamingBank` — the primitive that lets a tile position's causal state leave
// memory and come back unchanged.
//
// 🔑 **Why this exists, and why it is necessary rather than an optimisation** (`GAP-PROGRAM.md` V12-B):
// the tiled temporal driver keeps ONE bank per spatial tile position, each ~0.765 GiB at a 256² tile,
// and every one of them must stay live between its uses. That makes bank residency `positions × 0.765
// GiB` — 6.9 GiB at a 512² output, 26.8 GiB at SD ×2, and **130 GiB at HD→4K**, which is one of the
// product's two named customer segments and fits on nothing we own. The banks are pure carry-over
// state, touched once per boundary, so the way past it is to keep only the active one resident.
//
// 🔑 **The format is `GranuleSpillFile` (BlockStreamKit's `GranuleIO` product), not safetensors,**
// because V12-B3 measured the safetensors round trip SERIALIZATION-bound: 59.7 GiB out + 59.7 GiB
// back in 108.6 s ≈ 1.27 GB/s effective against the same volume's 8.74 GB/s raw — ~7× under, from
// 57-tensors-per-bank framing + per-array host materialization — and that made eviction cost +48%
// wall clock. The spill file is the "contiguous single-buffer bank format" that receipt named:
// one aligned file per bank, payloads written straight from unified memory and pread straight back
// into it. Semantics here are UNCHANGED — same count sentinel, same refusal behaviour, same
// bit-identity contract; only the transport moved.
//
// ⚠️ `SEEDVR2_BANK_STORE_LEGACY=1` restores the safetensors transport. It exists so the composite
// cost of the format itself can be A/B-measured through the shipping surface (the V12-B3 lesson:
// a ratio of two separately-measured quantities is not a measurement) — it is not a product switch.
//
// ⚠️ **Offloading is necessary in EITHER loop ordering, and by itself it is not sufficient.** Under the
// driver's current chunk-major iteration every bank round-trips once per chunk (~9 frames), which at 4K
// is ~29 GiB/frame of I/O against ~3.3 s/frame of compute. Tile-major iteration amortises the same
// traffic over a segment (~64 frames) instead, ≈7× less. This file is the half that is orthogonal to
// that restructure — it is needed either way, so it lands first and on its own.
//
// ⚠️ **This only moves `phys_footprint` because the engine caps the MLX buffer cache.** Dropping the
// last reference to a tail returns its buffer to MLX's pool, not to the OS; the memory is reclaimed
// only once the pool is over its cap. `PORT-QUEUE.md` cross-cutting note 3 — *set `GPU.set(cacheLimit:)`,
// never `memoryLimit`* — is what makes eviction real rather than bookkeeping.
import Foundation
import GranuleIO
import MLX

extension VAEStreamingBank {

    /// Sentinel key holding the tail COUNT, so `nil` slots survive the round trip.
    ///
    /// 🔑 Positions are load-bearing: `adoptStreamingMemory` re-installs tails *positionally* onto
    /// `causalConvs` and preconditions on the count. A bank whose trailing nils were dropped would
    /// fail that precondition, and one whose interior nils collapsed would silently install every
    /// tail onto the wrong conv — which is why the count is stored explicitly rather than inferred
    /// from the highest index present. (Legacy transport: a 1-element sentinel tensor. Spill
    /// transport: the same key in the file's `userInfo` table.)
    private static let countKey = "__tail_count"

    private static func key(_ index: Int) -> String { "t\(index)" }

    /// Measurement-only escape back to the safetensors transport (see the header). Read once —
    /// both directions of one process must speak the same format, or the refusal gate weakens.
    static let legacyTransport =
        ProcessInfo.processInfo.environment["SEEDVR2_BANK_STORE_LEGACY"] == "1"

    /// Write this bank to `url` (contiguous granule spill file). Non-nil tails only; the count
    /// restores the shape.
    ///
    /// ⚠️ **Evaluates the tails.** They are usually already settled — the driver exports after an
    /// `eval` — but a lazily-built tail would otherwise be written as an unevaluated graph.
    public func write(to url: URL) throws {
        var tensors: [(key: String, array: MLXArray)] = []
        for (i, tail) in tails.enumerated() {
            guard let tail else { continue }
            tensors.append((Self.key(i), tail))
        }
        if Self.legacyTransport {
            eval(tensors.map(\.array))
            var arrays = Dictionary(uniqueKeysWithValues: tensors)
            arrays[Self.countKey] = MLXArray([Int32(tails.count)])
            try save(arrays: arrays, url: url)
        } else {
            try GranuleSpillFile.write(
                tensors: tensors,
                userInfo: [Self.countKey: String(tails.count)],
                to: url)
        }
    }

    /// Read a bank previously written by ``write(to:)``.
    ///
    /// - Throws: ``VAEStreamingBankStoreError/malformed(_:)`` if the file is not a spill file, the
    ///   count sentinel is missing, or a stored index is out of range or duplicated — a corrupt or
    ///   foreign file, which must not be adopted into a VAE where it would be installed positionally
    ///   onto the wrong convolutions.
    public static func read(from url: URL) throws -> VAEStreamingBank {
        let named: [(String, MLXArray)]
        let countString: String?
        if legacyTransport {
            let arrays = try loadArrays(url: url)
            named = arrays.filter { $0.key != countKey }.map { ($0.key, $0.value) }
            countString = arrays[countKey].map { String($0.item(Int32.self)) }
        } else {
            let contents: GranuleSpillFile.Contents
            do {
                contents = try GranuleSpillFile.read(from: url)
            } catch let error as GranuleIOError {
                // Structural refusal (bad magic/version/table) — OUR gate, one error surface.
                throw VAEStreamingBankStoreError.malformed(String(describing: error))
            }
            named = contents.tensors
            countString = contents.userInfo[countKey]
        }
        guard let countString, let count = Int(countString) else {
            throw VAEStreamingBankStoreError.malformed("missing \(countKey) — not a streaming bank")
        }
        guard count >= 0 else {
            throw VAEStreamingBankStoreError.malformed("negative tail count \(count)")
        }
        var tails = [MLXArray?](repeating: nil, count: count)
        for (name, array) in named {
            guard name.hasPrefix("t"), let index = Int(name.dropFirst()) else {
                throw VAEStreamingBankStoreError.malformed("unexpected key \(name)")
            }
            guard index >= 0, index < count else {
                throw VAEStreamingBankStoreError.malformed(
                    "tail index \(index) outside 0..<\(count) — the file does not match its own count")
            }
            guard tails[index] == nil else {
                throw VAEStreamingBankStoreError.malformed("duplicate tail \(name)")
            }
            tails[index] = array
        }
        return VAEStreamingBank(tails: tails)
    }

    /// Total retained bytes, for sizing and for logging what an eviction actually reclaimed.
    public var byteCount: Int {
        tails.reduce(0) { $0 + ($1?.nbytes ?? 0) }
    }
}

public enum VAEStreamingBankStoreError: Error, CustomStringConvertible {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let why): return "streaming bank file is malformed: \(why)"
        }
    }
}
