import Foundation
import MLXToolKit
import SeedVR2MLX

// MARK: - Where the weights come from (contract 1.24: the ENGINE materializes, load() just loads)

extension SeedVR2Configuration {
    /// Below this stamped headroom an fp16 configuration loads the near-lossless int8 repo instead
    /// (`BudgetAware`): fp16 needs ~7.5 GB resident + a multi-GB transient (≈11 GB working set).
    static let fp16MinBudgetBytes: UInt64 = 11_000_000_000

    /// The quant `load()` materializes and loads: `quant`, except that an fp16 configuration stamped
    /// with less than ~11 GB of headroom drops to int8 (near-lossless, ~50.3 dB). The engine stamps the
    /// budget BEFORE its materialization pass, so the declared source follows the same substitution — a
    /// tight machine never downloads the 8.4 GB fp16 repo it would not load.
    public var effectiveQuant: Quant {
        if quant == .fp16, let budget = availableBudgetBytes, budget < Self.fp16MinBudgetBytes { return .int8 }
        return quant
    }

    /// The repo `load()` reads: `repoOverride` when set, else the canonical repo for `effectiveQuant`.
    public var effectiveRepo: String { repoOverride ?? Self.repo(for: effectiveQuant) }

    /// Every file a SeedVR2 weights repo carries that `load()` reads — and all a source fetches (the
    /// repos' README and .gitattributes are not weights).
    public static let weightFiles: [String] = HFHub.seedvr2Files

    /// `<root>/seedvr2-mlx/<org>--<name>/` — where v0.9.x and earlier downloaded whenever the engine
    /// stamped a store root: the core's own downloader, beside the store's `models--…` layout rather
    /// than in it. Read (and adopted) for compatibility only; nothing new is ever written here.
    public static func legacyStoreDirectory(storeRoot: URL, repo: String) -> URL {
        storeRoot.appending(path: "seedvr2-mlx", directoryHint: .isDirectory)
            .appending(path: repo.replacingOccurrences(of: "/", with: "--"), directoryHint: .isDirectory)
    }

    /// Where this configuration's weights already are, without changing anything on disk: the explicit
    /// `snapshotDirectory` when set (and nothing else); else, under `storeRoot`, the engine's flat layout
    /// (where its materializer lands files), a hub-client snapshot, or the v0.9.x in-store layout.
    /// `nil` ⇒ the source is missing. One function answers both the engine's missing-probe and what
    /// `load()` reads, so the two can never disagree.
    ///
    /// The store-less core cache (`~/Library/Caches/seedvr2-mlx`) is deliberately NOT a candidate: with
    /// a store attached, v0.9.x never read it either (it downloaded into the store), and counting it
    /// would leave weights outside the store the user picked.
    public func existingWeightsDirectory(storeRoot: URL?) -> URL? {
        if let snapshotDirectory { return Self.isComplete(snapshotDirectory) ? snapshotDirectory : nil }
        guard let storeRoot else { return nil }
        let store = ModelStore(root: storeRoot)
        let candidates = [store.directory(for: effectiveRepo),
                          store.snapshotDirectory(for: effectiveRepo, revision: "main"),
                          Self.legacyStoreDirectory(storeRoot: storeRoot, repo: effectiveRepo)]
        return candidates.compactMap { $0 }.first(where: Self.isComplete)
    }

    /// Every weight file present with a non-zero size. Presence, not integrity — the same contract as
    /// the engine's probe. (An in-flight download lives at `<file>.part`, so it never counts.)
    static func isComplete(_ directory: URL) -> Bool {
        weightFiles.allSatisfy { file in
            let attributes = try? FileManager.default.attributesOfItem(
                atPath: directory.appending(path: file, directoryHint: .notDirectory).path)
            return ((attributes?[.size] as? NSNumber)?.intValue ?? 0) > 0
        }
    }

    /// The directory `load()` reads, fetching or moving only what the engine's pass left undone:
    ///
    /// 1. explicit `snapshotDirectory` — as-is; never touches the network;
    /// 2. no store root — the core's own cache (`~/Library/Caches/seedvr2-mlx/<org>--<name>`),
    ///    downloading there if absent: the v0.9.x store-less behaviour, unchanged;
    /// 3. store root — the store copy. A v0.9.x in-store snapshot is ADOPTED into the flat layout by
    ///    rename (see `adoptLegacySnapshot`); if nothing is there at all (a caller that constructed the
    ///    package directly, bypassing the engine's materialization pass), the package downloads into
    ///    the flat layout itself so the next engine pass finds it.
    func materializedWeightsDirectory() throws -> URL {
        if let snapshotDirectory { return snapshotDirectory }
        let repo = effectiveRepo
        guard let root = modelsRootDirectory else {
            return try HFHub.snapshot(repoId: repo, files: Self.weightFiles)
        }
        guard let flat = ModelStore(root: root).directory(for: repo) else {
            return try HFHub.snapshot(repoId: repo, files: Self.weightFiles)
        }
        if let existing = existingWeightsDirectory(storeRoot: root) {
            let legacy = Self.legacyStoreDirectory(storeRoot: root, repo: repo)
            guard existing.standardizedFileURL.path == legacy.standardizedFileURL.path else { return existing }
            return Self.adoptLegacySnapshot(legacy, into: flat)
        }
        return try HFHub.snapshot(repoId: repo, files: Self.weightFiles, cacheDir: flat)
    }

    /// Moves a complete v0.9.x in-store snapshot into the store's flat layout (`models--<org>--<name>/`,
    /// where the engine already writes this repo's install marker), then removes the emptied legacy
    /// directories. Both live under the same store root, so each move is a rename: no bytes are copied
    /// and nothing is downloaded. Returns the directory to load from — the flat one on success, the
    /// legacy one if the volumes differ or any move fails (the moves already made are put back first).
    ///
    /// Why adopt rather than read in place: in place, the storage panel would credit the repo's marker
    /// with an empty directory, and removing the model from the store would leave ~4.7 GB behind.
    static func adoptLegacySnapshot(_ legacy: URL, into flat: URL) -> URL {
        let fm = FileManager.default
        // Across volumes a move is a multi-GB copy — load in place instead.
        guard sameVolume(legacy, flat.deletingLastPathComponent()) else { return legacy }
        do {
            try fm.createDirectory(at: flat, withIntermediateDirectories: true)
        } catch {
            return legacy
        }
        var moved: [String] = []
        for file in weightFiles {
            let source = legacy.appending(path: file, directoryHint: .notDirectory)
            let destination = flat.appending(path: file, directoryHint: .notDirectory)
            do {
                // The flat layout is incomplete (else it would have been read first), so a file already
                // there is a leftover of an interrupted download — the legacy copy replaces it.
                if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
                try fm.moveItem(at: source, to: destination)
                moved.append(file)
            } catch {
                for file in moved {
                    try? fm.moveItem(at: flat.appending(path: file, directoryHint: .notDirectory),
                                     to: legacy.appending(path: file, directoryHint: .notDirectory))
                }
                return legacy
            }
        }
        removeIfEmpty(legacy)
        removeIfEmpty(legacy.deletingLastPathComponent())   // `<root>/seedvr2-mlx`, if this was its last repo
        return flat
    }

    private static func sameVolume(_ a: URL, _ b: URL) -> Bool {
        let key: Set<URLResourceKey> = [.volumeIdentifierKey]
        guard let va = try? a.resourceValues(forKeys: key).volumeIdentifier,
              let vb = try? b.resourceValues(forKeys: key).volumeIdentifier else { return false }
        return va.isEqual(vb)
    }

    private static func removeIfEmpty(_ directory: URL) {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: directory.path),
              entries.allSatisfy({ $0 == ".DS_Store" }) else { return }
        try? fm.removeItem(at: directory)
    }
}

/// Fresh-machine sources (contract 1.24: the ENGINE downloads them into the store before `load()`, with
/// its `.downloading` phase). One role per quant lane, each matching only that repo's four weight files —
/// an int8 configuration never pulls the 8.4 GB fp16 transformer.
extension SeedVR2Configuration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: effectiveQuant == .fp16 ? "seedvr2-fp16" : "seedvr2-int8",
                      repo: effectiveRepo, revision: "main", matching: Self.weightFiles)]
    }

    /// Explicit `snapshotDirectory` first (complete → nothing missing; incomplete → the lane is missing),
    /// then the store: the flat layout, a hub snapshot, or a v0.9.x in-store snapshot — the last so an
    /// existing user is never re-downloaded (`load()` adopts it into the flat layout).
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        existingWeightsDirectory(storeRoot: storeRoot) == nil ? weightSources : []
    }
}
