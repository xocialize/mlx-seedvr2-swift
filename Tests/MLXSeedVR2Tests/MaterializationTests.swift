// MaterializationTests.swift — the offline MAT gate (engine ≥ 0.19.0; contract 1.24 engine-executed
// materialization) for SeedVR2, plus the store layouts `load()` reads. MAT-1 store-stampable, MAT-2 a
// source declared, MAT-3 role/repo hygiene, MAT-4 fresh-machine posture (nil store ⇒ the lane is missing),
// MAT-5 an explicit snapshot satisfies — per selectable quant lane, because the declaration changes with
// `quant` (quant-as-repo). Weightless: tiny probe files stand in for the four weight files.
//
// The v0.9.x compatibility rule pinned here: a snapshot at `<root>/seedvr2-mlx/<org>--<name>/` (where
// v0.9.x downloaded whenever a store root was stamped) satisfies the source — the engine must NOT
// re-download ~4.7 GB — and `load()`'s resolver adopts it into the flat layout by rename.

import Foundation
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import Testing
@testable import MLXSeedVR2

struct MaterializationTests {

    private static let int8Repo = "mlx-community/SeedVR2-3B-mlx-int8"
    private static let fp16Repo = "mlx-community/SeedVR2-3B-mlx"

    private func temporaryDirectory(_ tag: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("seedvr2-\(tag)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Writes probe weight files into `dir` (all four, or all but the last).
    private func populate(_ dir: URL, complete: Bool = true, byte: UInt8 = 1) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let files = complete ? SeedVR2Configuration.weightFiles : Array(SeedVR2Configuration.weightFiles.dropLast())
        for file in files {
            #expect(FileManager.default.createFile(atPath: dir.appendingPathComponent(file).path,
                                                   contents: Data([byte])))
        }
    }

    private func contents(_ dir: URL) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
    }

    // MARK: - The MAT gate, per quant lane

    @Test(arguments: [Quant.int8, .fp16])
    func fullMATGatePassesPerLane(quant: Quant) throws {
        let dir = try temporaryDirectory("mat")
        defer { try? FileManager.default.removeItem(at: dir) }
        try populate(dir)
        let report = MaterializationConformance.check(
            freshConfiguration: SeedVR2Configuration(quant: quant),
            satisfiedConfiguration: SeedVR2Configuration(quant: quant, snapshotDirectory: dir))
        #expect(report.passed, "\(quant):\n\(report.summary)")
    }

    @Test func oneRolePerQuantMatchingOnlyTheWeightFiles() {
        let int8 = SeedVR2Configuration(quant: .int8).weightSources
        let fp16 = SeedVR2Configuration(quant: .fp16).weightSources
        #expect(int8 == [WeightSource(role: "seedvr2-int8", repo: Self.int8Repo, revision: "main",
                                      matching: ["config.json", "pos_emb.safetensors", "vae.safetensors",
                                                 "transformer.safetensors"])])
        #expect(fp16.map(\.role) == ["seedvr2-fp16"])
        #expect(fp16.map(\.repo) == [Self.fp16Repo])
        // The repos' README / .gitattributes are not weights — exact names, no wildcard.
        #expect(int8.allSatisfy { ($0.matching ?? []).allSatisfy { !$0.contains("*") } })
    }

    /// BudgetAware: the engine stamps the budget before its materialization pass, so a tight fp16
    /// configuration declares — and therefore downloads — the int8 repo it will actually load.
    @Test func tightBudgetFp16DeclaresTheInt8Source() {
        var cfg = SeedVR2Configuration(quant: .fp16)
        cfg.availableBudgetBytes = 8_000_000_000
        #expect(cfg.effectiveQuant == .int8)
        #expect(cfg.weightSources.map(\.repo) == [Self.int8Repo])
        #expect(cfg.weightSources.map(\.role) == ["seedvr2-int8"])
        cfg.availableBudgetBytes = 16_000_000_000
        #expect(cfg.effectiveQuant == .fp16)
        #expect(cfg.weightSources.map(\.repo) == [Self.fp16Repo])
        #expect(cfg.quant == .fp16, "the charged footprint (QuantConfigured) is the configured quant")
    }

    @Test func repoOverrideIsDeclaredAndResolved() throws {
        let root = try temporaryDirectory("override")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = SeedVR2Configuration(repoOverride: "acme/SeedVR2-custom", modelsRootDirectory: root)
        #expect(cfg.weightSources.map(\.repo) == ["acme/SeedVR2-custom"])
        let flat = root.appendingPathComponent("models--acme--SeedVR2-custom")
        try populate(flat)
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
        #expect(try cfg.materializedWeightsDirectory().standardizedFileURL.path == flat.standardizedFileURL.path)
    }

    // MARK: - Store layouts the source is satisfied by (and load() reads)

    /// The engine's flat layout (contract 1.24): satisfied only when EVERY weight file is there.
    @Test func storeFlatLayoutSatisfiesOnlyWhenComplete() throws {
        let root = try temporaryDirectory("flat")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = SeedVR2Configuration(modelsRootDirectory: root)
        let flat = try #require(ModelStore(root: root).directory(for: Self.int8Repo))
        #expect(cfg.missingWeightSources(storeRoot: root).map(\.role) == ["seedvr2-int8"], "empty store ⇒ missing")

        // A marker-only directory (what the engine stamps after any load) is not a materialized model.
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: flat.appendingPathComponent(ModelStore.markerName).path,
                                       contents: Data("{}".utf8))
        #expect(cfg.missingWeightSources(storeRoot: root).count == 1, "marker only ⇒ missing")

        try populate(flat, complete: false)
        #expect(cfg.missingWeightSources(storeRoot: root).count == 1, "one file short ⇒ still missing")
        try populate(flat)
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
        #expect(try cfg.materializedWeightsDirectory().standardizedFileURL.path == flat.standardizedFileURL.path)

        // An empty (0-byte) file is an interrupted write, not a weight.
        FileManager.default.createFile(atPath: flat.appendingPathComponent("vae.safetensors").path, contents: Data())
        #expect(cfg.missingWeightSources(storeRoot: root).count == 1, "0-byte file ⇒ missing")
    }

    /// A hub-client snapshot (`refs/main` → `snapshots/<commit>/`) satisfies the source too.
    @Test func hubSnapshotLayoutSatisfies() throws {
        let root = try temporaryDirectory("hub")
        defer { try? FileManager.default.removeItem(at: root) }
        let repoDir = try #require(ModelStore(root: root).directory(for: Self.int8Repo))
        let snapshot = repoDir.appendingPathComponent("snapshots/abc123")
        try populate(snapshot)
        try FileManager.default.createDirectory(at: repoDir.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try "abc123".write(to: repoDir.appendingPathComponent("refs/main"), atomically: true, encoding: .utf8)
        let cfg = SeedVR2Configuration(modelsRootDirectory: root)
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
        #expect(try cfg.materializedWeightsDirectory().standardizedFileURL.path == snapshot.standardizedFileURL.path)
    }

    /// The explicit snapshot wins over the store and never reads as satisfied unless complete.
    @Test func explicitSnapshotWinsOverTheStore() throws {
        let root = try temporaryDirectory("explicit")
        defer { try? FileManager.default.removeItem(at: root) }
        let flat = try #require(ModelStore(root: root).directory(for: Self.int8Repo))
        try populate(flat)
        let snap = root.appendingPathComponent("my-snapshot")
        try populate(snap, complete: false)
        let cfg = SeedVR2Configuration(snapshotDirectory: snap, modelsRootDirectory: root)
        #expect(cfg.missingWeightSources(storeRoot: root).count == 1, "incomplete explicit snapshot ⇒ missing")
        #expect(try cfg.materializedWeightsDirectory() == snap, "explicit wins — load() reads it, as-is")
        try populate(snap)
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
    }

    // MARK: - The existing-cache decision: honor a v0.9.x in-store snapshot, adopt it by rename

    /// Mirrors the real Forge store after v0.9.x: the weights under `seedvr2-mlx/<org>--<name>/` and a
    /// marker-only `models--…` directory the engine stamped after load. The source must read as
    /// satisfied (no re-download), and the resolver must MOVE the files into the flat layout beside the
    /// marker, then remove the emptied legacy directories.
    @Test func legacyInStoreSnapshotIsHonoredThenAdopted() async throws {
        let root = try temporaryDirectory("legacy")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = SeedVR2Configuration.legacyStoreDirectory(storeRoot: root, repo: Self.int8Repo)
        #expect(legacy.path.hasSuffix("/seedvr2-mlx/mlx-community--SeedVR2-3B-mlx-int8"))
        try populate(legacy, byte: 7)
        let flat = try #require(ModelStore(root: root).directory(for: Self.int8Repo))
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: flat.appendingPathComponent(ModelStore.markerName).path,
                                       contents: Data("{}".utf8))

        let cfg = SeedVR2Configuration(modelsRootDirectory: root)
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty, "a v0.9.x snapshot must not be re-downloaded")

        // The engine agrees (its needsDownload is the app's first-run routing signal).
        let engine = MLXServeEngine()
        await engine.useModelStore(ModelStore(root: root))
        _ = try await engine.register(SeedVR2UpscalePackage.registration, configuration: SeedVR2Configuration())
        #expect(await engine.needsDownload(.imageUpscale) == false)

        let resolved = try cfg.materializedWeightsDirectory()
        #expect(resolved.standardizedFileURL.path == flat.standardizedFileURL.path)
        #expect(contents(flat) == Set(SeedVR2Configuration.weightFiles + [ModelStore.markerName]))
        // Moved, not copied: the bytes are the legacy files' bytes.
        #expect(try Data(contentsOf: flat.appendingPathComponent("transformer.safetensors")) == Data([7]))
        #expect(!FileManager.default.fileExists(atPath: legacy.path), "emptied legacy repo dir removed")
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("seedvr2-mlx").path),
                "emptied seedvr2-mlx dir removed")
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty, "still satisfied after adoption")
        // Idempotent: a second resolve reads the flat layout and moves nothing.
        #expect(try cfg.materializedWeightsDirectory().standardizedFileURL.path == flat.standardizedFileURL.path)
    }

    /// A complete flat layout wins; a leftover legacy copy is read by nobody and left untouched.
    @Test func completeFlatLayoutWinsOverLegacy() throws {
        let root = try temporaryDirectory("both")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = SeedVR2Configuration.legacyStoreDirectory(storeRoot: root, repo: Self.int8Repo)
        try populate(legacy, byte: 7)
        let flat = try #require(ModelStore(root: root).directory(for: Self.int8Repo))
        try populate(flat, byte: 9)
        let cfg = SeedVR2Configuration(modelsRootDirectory: root)
        #expect(try cfg.materializedWeightsDirectory().standardizedFileURL.path == flat.standardizedFileURL.path)
        #expect(contents(legacy) == Set(SeedVR2Configuration.weightFiles), "legacy untouched")
        #expect(try Data(contentsOf: flat.appendingPathComponent("vae.safetensors")) == Data([9]))
    }

    /// A partial flat layout (an interrupted engine download) is replaced by the complete legacy files.
    @Test func legacyAdoptionReplacesAPartialFlatLayout() throws {
        let root = try temporaryDirectory("partial")
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = SeedVR2Configuration.legacyStoreDirectory(storeRoot: root, repo: Self.int8Repo)
        try populate(legacy, byte: 7)
        let flat = try #require(ModelStore(root: root).directory(for: Self.int8Repo))
        try populate(flat, complete: false, byte: 3)
        let cfg = SeedVR2Configuration(modelsRootDirectory: root)
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
        #expect(try cfg.materializedWeightsDirectory().standardizedFileURL.path == flat.standardizedFileURL.path)
        for file in SeedVR2Configuration.weightFiles {
            #expect(try Data(contentsOf: flat.appendingPathComponent(file)) == Data([7]), "\(file) from legacy")
        }
        #expect(!FileManager.default.fileExists(atPath: legacy.path))
    }

    /// If a move fails, the resolver puts back what it moved and loads the legacy snapshot in place —
    /// never a copy, never a download.
    @Test func failedAdoptionLoadsLegacyInPlace() throws {
        let root = try temporaryDirectory("readonly")
        let flat = try #require(ModelStore(root: root).directory(for: Self.int8Repo))
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: flat.path)
            try? FileManager.default.removeItem(at: root)
        }
        let legacy = SeedVR2Configuration.legacyStoreDirectory(storeRoot: root, repo: Self.int8Repo)
        try populate(legacy, byte: 7)
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: flat.path)

        let cfg = SeedVR2Configuration(modelsRootDirectory: root)
        let resolved = try cfg.materializedWeightsDirectory()
        #expect(resolved.standardizedFileURL.path == legacy.standardizedFileURL.path)
        #expect(contents(legacy) == Set(SeedVR2Configuration.weightFiles), "legacy intact")
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty, "still honored next time")
    }

    // MARK: - Engine routing on a fresh registration

    /// Register is offline — with an empty store the source reads as needing a download.
    @Test func engineNeedsDownloadOnAFreshStore() async throws {
        let root = try temporaryDirectory("fresh")
        defer { try? FileManager.default.removeItem(at: root) }
        let engine = MLXServeEngine()
        await engine.useModelStore(ModelStore(root: root))
        _ = try await engine.register(SeedVR2UpscalePackage.registration, configuration: SeedVR2Configuration())
        #expect(await engine.needsDownload(.imageUpscale))
        #expect(await engine.needsDownload(.videoUpscale), "one source backs both surfaces")
    }
}
