// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// `temporalWindow` persistence (V12-D): the CodingKeys discipline — portable knobs round-trip,
// pre-0.8.0 configs (no `temporalWindow` key) decode to the default instead of failing.
import Foundation
import XCTest

@testable import MLXSeedVR2

final class ConfigurationCodingTests: XCTestCase {

    func testTemporalWindowRoundTrips() throws {
        var config = SeedVR2Configuration()
        XCTAssertEqual(config.temporalWindow, 9, "default is the 16 GB operating point")
        config.temporalWindow = 13
        let decoded = try JSONDecoder().decode(SeedVR2Configuration.self,
                                               from: JSONEncoder().encode(config))
        XCTAssertEqual(decoded.temporalWindow, 13)
        XCTAssertEqual(decoded.quant, config.quant)
        XCTAssertEqual(decoded.tileSize, config.tileSize)
    }

    func testLegacyConfigWithoutTemporalWindowDecodes() throws {
        // A v0.7.x persisted config — no temporalWindow key.
        let legacy = """
        {"quant":"int8","seed":0,"colorCorrect":true,"defaultScale":2,
         "tileSize":256,"tileOverlap":32,"tileHalo":0,"imageWholeFramePixels":1048576}
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(SeedVR2Configuration.self, from: legacy)
        XCTAssertEqual(decoded.temporalWindow, 9)
        XCTAssertEqual(decoded.tileSize, 256)
    }

    func testEnvironmentFieldsStayUnpersisted() throws {
        var config = SeedVR2Configuration()
        config.snapshotDirectory = URL(fileURLWithPath: "/tmp/x")
        config.modelsRootDirectory = URL(fileURLWithPath: "/tmp/y")
        config.availableBudgetBytes = 123
        let decoded = try JSONDecoder().decode(SeedVR2Configuration.self,
                                               from: JSONEncoder().encode(config))
        XCTAssertNil(decoded.snapshotDirectory)
        XCTAssertNil(decoded.modelsRootDirectory)
        XCTAssertNil(decoded.availableBudgetBytes)
    }
}
