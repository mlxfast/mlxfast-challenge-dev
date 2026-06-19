import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastSubmission

@Test
func submissionPackagesEditableFilesAndSkipsMacMetadata() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "local metadata".write(
        to: sources.appendingPathComponent(".DS_Store"),
        atomically: true,
        encoding: .utf8
    )
    try "appledouble".write(
        to: sources.appendingPathComponent("._A.swift"),
        atomically: true,
        encoding: .utf8
    )
    let macosx = sources.appendingPathComponent("__MACOSX", isDirectory: true)
    try FileManager.default.createDirectory(at: macosx, withIntermediateDirectories: true)
    try "ignored".write(
        to: macosx.appendingPathComponent("Ignored.swift"),
        atomically: true,
        encoding: .utf8
    )

    let archive = root.appendingPathComponent("submission.zip")
    let report = try SubmissionSupport.packageEditablePaths(
        contractPath: root.appendingPathComponent("benchmark.json").path,
        outputPath: archive.path
    )

    #expect(report.fileCount == 1)
    #expect(report.byteCount == "public enum A {}\n".utf8.count)
    #expect(report.archiveSha256.count == 64)
    #expect(FileManager.default.fileExists(atPath: archive.path))
}

@Test
func submissionRejectsGeneratedArtifactsInEditablePaths() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["weights"])
    try FileManager.default.createDirectory(
        at: root.appendingPathComponent("weights", isDirectory: true),
        withIntermediateDirectories: true
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsDuplicateEditablePaths() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources", "Sources"])

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsOverlappingEditablePathsThatSelectSameFile() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources", "Sources/A.swift"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsBackslashContractPaths() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources\\A.swift"])

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsSymlinkInsideEditableDirectory() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    let target = sources.appendingPathComponent("A.swift")
    try "public enum A {}\n".write(to: target, atomically: true, encoding: .utf8)
    try FileManager.default.createSymbolicLink(
        at: sources.appendingPathComponent("Alias.swift"),
        withDestinationURL: target
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.ensureWorkspace(
            contractPath: root.appendingPathComponent("benchmark.json").path
        )
    }
}

@Test
func submissionRejectsArchiveOutputInsideEditablePath() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.packageEditablePaths(
            contractPath: root.appendingPathComponent("benchmark.json").path,
            outputPath: sources.appendingPathComponent("submission.zip").path
        )
    }
}

@Test
func submissionRejectsNonZipArchiveOutput() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.packageEditablePaths(
            contractPath: root.appendingPathComponent("benchmark.json").path,
            outputPath: root.appendingPathComponent("submission.txt").path
        )
    }
}

@Test
func submissionRejectsSourceByteCountAboveLimit() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "1234".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.packageEditablePaths(
            contractPath: root.appendingPathComponent("benchmark.json").path,
            outputPath: root.appendingPathComponent("submission.zip").path,
            maxByteCount: 3
        )
    }
}

@Test
func submissionAllowsUnlimitedSourceByteCountWhenExplicitlyConfigured() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "1234".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )

    let report = try SubmissionSupport.packageEditablePaths(
        contractPath: root.appendingPathComponent("benchmark.json").path,
        outputPath: root.appendingPathComponent("submission.zip").path,
        maxByteCount: nil
    )

    #expect(report.byteCount == 4)
    #expect(report.archiveSha256.count == 64)
}

@Test
func submissionPackagesEditablePathsAsTarGzipForYukon() throws {
    let root = try makeSubmissionWorkspace(editablePaths: ["Sources"])
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try "public enum A {}\n".write(
        to: sources.appendingPathComponent("A.swift"),
        atomically: true,
        encoding: .utf8
    )
    let archive = root.appendingPathComponent("submission.tar.gz")

    let report = try SubmissionSupport.packageEditablePathsTarGzip(
        contractPath: root.appendingPathComponent("benchmark.json").path,
        outputPath: archive.path
    )

    #expect(report.fileCount == 1)
    #expect(report.byteCount == "public enum A {}\n".utf8.count)
    #expect(report.archivePath == archive.path)
    #expect(report.archiveSha256.count == 64)
    #expect(FileManager.default.fileExists(atPath: archive.path))
}

@Test
func loginStoresYukonCompatibleCredentialsFile() throws {
    let home = try temporarySubmissionDirectory()
    let path = try SubmissionSupport.storeCredentials(
        apiKey: "  test-key  ",
        apiBaseURL: "https://yukon.example.test/",
        homeDirectory: home,
        environment: [:],
        storedAt: 123.5
    )
    let expected = home
        .appendingPathComponent(".config", isDirectory: true)
        .appendingPathComponent("mlxfast", isDirectory: true)
        .appendingPathComponent("credentials")

    #expect(path == expected.path)

    let credentials = try SubmissionSupport.loadCredentials(homeDirectory: home, environment: [:])
    #expect(
        credentials == StoredCredentials(
            apiKey: "test-key",
            apiBaseURL: "https://yukon.example.test",
            storedAt: 123.5
        )
    )

    let attributes = try FileManager.default.attributesOfItem(atPath: expected.path)
    let permissions = try #require(attributes[.posixPermissions] as? NSNumber)
    #expect(permissions.intValue & 0o777 == 0o600)
}

@Test
func loginHonorsXDGConfigHomeForCredentialPath() throws {
    let root = try temporarySubmissionDirectory()
    let configHome = root.appendingPathComponent("xdg-config", isDirectory: true)
    let path = try SubmissionSupport.storeCredentials(
        apiKey: "test-key",
        homeDirectory: root,
        environment: ["XDG_CONFIG_HOME": configHome.path],
        storedAt: 1
    )

    let expected = configHome
        .appendingPathComponent("mlxfast", isDirectory: true)
        .appendingPathComponent("credentials")
    #expect(path == expected.path)
    let credentials = try SubmissionSupport.loadCredentials(
        homeDirectory: root,
        environment: ["XDG_CONFIG_HOME": configHome.path]
    )
    #expect(credentials.apiKey == "test-key")
}

@Test
func loginRejectsRelativeConfigHome() throws {
    let home = try temporarySubmissionDirectory()

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.storeCredentials(
            apiKey: "test-key",
            homeDirectory: home,
            environment: ["XDG_CONFIG_HOME": "relative/config"]
        )
    }
}

@Test
func loginLoadsLegacySwiftCredentialsJson() throws {
    let home = try temporarySubmissionDirectory()
    let legacyPath = try SubmissionSupport.legacyCredentialsPath(homeDirectory: home, environment: [:])
    try FileManager.default.createDirectory(
        at: legacyPath.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try """
    {
      "api_key": "legacy-key",
      "stored_at": 22
    }
    """.write(to: legacyPath, atomically: true, encoding: .utf8)

    let credentials = try SubmissionSupport.loadCredentials(homeDirectory: home, environment: [:])
    #expect(credentials == StoredCredentials(apiKey: "legacy-key", storedAt: 22))
}

@Test
func loginRejectsEmptyAPIKey() throws {
    let home = try temporarySubmissionDirectory()

    #expect(throws: MLXFastError.self) {
        _ = try SubmissionSupport.storeCredentials(apiKey: "  \n", homeDirectory: home, environment: [:])
    }
}

@Test
func yukonClientMeUsesBearerAuth() throws {
    var sawRequest = false
    let client = try YukonClient(apiBaseURL: "https://yukon.example.test", apiKey: "secret") { request in
        sawRequest = true
        #expect(request.httpMethod == "GET")
        #expect(request.url?.absoluteString == "https://yukon.example.test/api/me")
        #expect(request.value(forHTTPHeaderField: "authorization") == "Bearer secret")
        return try httpResponse(
            for: request,
            statusCode: 200,
            body: """
            {
              "account": {
                "id": "acct-1",
                "email": "solver@example.test",
                "username": "solver"
              }
            }
            """
        )
    }

    let response = try client.me()

    #expect(sawRequest)
    #expect(response.account.email == "solver@example.test")
}

@Test
func yukonClientSubmitsRawGzipArchive() throws {
    let root = try temporarySubmissionDirectory()
    let archive = root.appendingPathComponent("submission.tar.gz")
    let archiveBytes = Data([1, 2, 3, 4])
    try archiveBytes.write(to: archive)
    var sawRequest = false
    let client = try YukonClient(apiBaseURL: "https://yukon.example.test/base", apiKey: "secret") { request in
        sawRequest = true
        #expect(request.httpMethod == "POST")
        #expect(
            request.url?.absoluteString ==
                "https://yukon.example.test/base/api/benchmarks/bench%2Fname/submissions?claimedScore=1.25"
        )
        #expect(request.value(forHTTPHeaderField: "authorization") == "Bearer secret")
        #expect(request.value(forHTTPHeaderField: "idempotency-key") == "idem-1")
        #expect(request.value(forHTTPHeaderField: "content-type") == "application/gzip")
        #expect(request.httpBody == archiveBytes)
        return try httpResponse(
            for: request,
            statusCode: 202,
            body: submissionResponseJSON
        )
    }

    let response = try client.createSubmission(
        YukonSubmissionOptions(
            benchmark: "bench/name",
            archivePath: archive.path,
            idempotencyKey: "idem-1",
            claimedScore: 1.25
        )
    )

    #expect(sawRequest)
    #expect(response.submission.id == "sub-1")
    #expect(response.job?.id == "job-1")
}

@Test
func yukonClientSubmitsMultipartArchiveWhenNoteIsProvided() throws {
    let root = try temporarySubmissionDirectory()
    let archive = root.appendingPathComponent("submission.tar.gz")
    try Data([9, 8, 7]).write(to: archive)
    var sawRequest = false
    let client = try YukonClient(apiBaseURL: "https://yukon.example.test", apiKey: "secret") { request in
        sawRequest = true
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "content-type")?.hasPrefix("multipart/form-data; boundary=") == true)
        let body = try #require(request.httpBody)
        let text = String(data: body, encoding: .utf8) ?? ""
        #expect(text.contains("name=\"archive\"; filename=\"submission.tar.gz\""))
        #expect(text.contains("Content-Type: application/gzip"))
        #expect(text.contains("name=\"note\""))
        #expect(text.contains("changed streaming strategy"))
        return try httpResponse(
            for: request,
            statusCode: 202,
            body: submissionResponseJSON
        )
    }

    let response = try client.createSubmission(
        YukonSubmissionOptions(
            benchmark: "bench",
            archivePath: archive.path,
            idempotencyKey: "idem-2",
            note: " changed streaming strategy \n"
        )
    )

    #expect(sawRequest)
    #expect(response.submission.status == "received")
}

private func makeSubmissionWorkspace(editablePaths: [String]) throws -> URL {
    let root = try temporarySubmissionDirectory()
    let sources = root.appendingPathComponent("Sources", isDirectory: true)
    try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
    try writeSubmissionContract(
        root.appendingPathComponent("benchmark.json"),
        editablePaths: editablePaths
    )
    return root
}

private let submissionResponseJSON = """
{
  "submission": {
    "id": "sub-1",
    "benchmarkId": "bench",
    "status": "received",
    "note": null,
    "claimedScore": 1.25,
    "officialScore": null
  },
  "job": {
    "id": "job-1",
    "status": "queued"
  }
}
"""

private func httpResponse(
    for request: URLRequest,
    statusCode: Int,
    body: String
) throws -> (Data, HTTPURLResponse) {
    let url = try #require(request.url)
    let response = try #require(
        HTTPURLResponse(
            url: url,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: ["content-type": "application/json"]
        )
    )
    return (Data(body.utf8), response)
}

private func writeSubmissionContract(_ path: URL, editablePaths: [String]) throws {
    let encodedPaths = editablePaths
        .map { #""\#($0.replacingOccurrences(of: "\\", with: "\\\\"))""# }
        .joined(separator: ", ")
    let json = """
    {
      "schemaVersion": 1,
      "name": "fixture",
      "editablePaths": [\(encodedPaths)]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)
}

private func temporarySubmissionDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("mlxfast-submission-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
