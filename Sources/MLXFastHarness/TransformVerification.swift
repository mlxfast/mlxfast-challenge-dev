import CryptoKit
import Foundation
import MLXFastCore
import MLXFastTransform

public struct TransformVerificationOptions: Equatable {
    public let referencePath: String
    public let weightsPath: String
    public let temporaryParentPath: String?

    public init(referencePath: String, weightsPath: String, temporaryParentPath: String? = nil) {
        self.referencePath = referencePath
        self.weightsPath = weightsPath
        self.temporaryParentPath = temporaryParentPath
    }
}

public struct TransformVerificationReport: Codable, Equatable {
    public let referencePath: String
    public let weightsPath: String
    public let regeneratedPath: String
    public let fileCount: Int
    public let byteCount: Int
    public let sha256: String

    public init(
        referencePath: String,
        weightsPath: String,
        regeneratedPath: String,
        fileCount: Int,
        byteCount: Int,
        sha256: String
    ) {
        self.referencePath = referencePath
        self.weightsPath = weightsPath
        self.regeneratedPath = regeneratedPath
        self.fileCount = fileCount
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public enum TransformVerifier {
    public static func verify(_ options: TransformVerificationOptions) throws -> TransformVerificationReport {
        let weightsDirectory = URL(fileURLWithPath: options.weightsPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: weightsDirectory.path) else {
            throw MLXFastError.missingFile("weights directory not found at \(weightsDirectory.path)")
        }

        let parentDirectory = try verificationParent(
            explicitPath: options.temporaryParentPath,
            weightsDirectory: weightsDirectory
        )
        try FileManager.default.createDirectory(at: parentDirectory, withIntermediateDirectories: true)

        let regeneratedDirectory = parentDirectory.appendingPathComponent(
            ".mlxfast-transform-verify-\(UUID().uuidString)",
            isDirectory: true
        )
        defer {
            try? FileManager.default.removeItem(at: regeneratedDirectory)
        }

        let transformReport = try SwiftTransform.run(
            TransformOptions(referencePath: options.referencePath, outputPath: regeneratedDirectory.path)
        )
        let comparison = try compareDirectories(
            expected: regeneratedDirectory,
            actual: weightsDirectory
        )

        return TransformVerificationReport(
            referencePath: transformReport.referencePath,
            weightsPath: weightsDirectory.path,
            regeneratedPath: regeneratedDirectory.path,
            fileCount: comparison.fileCount,
            byteCount: comparison.byteCount,
            sha256: comparison.sha256
        )
    }

    private static func verificationParent(
        explicitPath: String?,
        weightsDirectory: URL
    ) throws -> URL {
        if let explicitPath, !explicitPath.isEmpty {
            return URL(fileURLWithPath: explicitPath).standardizedFileURL
        }
        let parent = weightsDirectory.deletingLastPathComponent()
        if parent.path.isEmpty {
            return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        }
        return parent
    }

    private struct Comparison: Equatable {
        let fileCount: Int
        let byteCount: Int
        let sha256: String
    }

    private static func compareDirectories(expected: URL, actual: URL) throws -> Comparison {
        let expectedFiles = try relativeRegularFiles(in: expected)
        let actualFiles = try relativeRegularFiles(in: actual)
        let expectedSet = Set(expectedFiles.keys)
        let actualSet = Set(actualFiles.keys)
        guard expectedSet == actualSet else {
            let missing = expectedSet.subtracting(actualSet).sorted()
            let extra = actualSet.subtracting(expectedSet).sorted()
            throw MLXFastError.invalidInput(
                "transform verification file set mismatch\(describeDifference(missing: missing, extra: extra))"
            )
        }

        var treeHasher = SHA256()
        var byteCount = 0
        for relativePath in expectedFiles.keys.sorted() {
            let expectedURL = expectedFiles[relativePath]!
            let actualURL = actualFiles[relativePath]!
            let expectedSize = try byteCountOfFile(expectedURL)
            let actualSize = try byteCountOfFile(actualURL)
            guard expectedSize == actualSize else {
                throw MLXFastError.invalidInput(
                    "transform verification mismatch for \(relativePath): expected \(expectedSize) bytes, found \(actualSize)"
                )
            }

            let digest = try compareFileBytes(
                expected: expectedURL,
                actual: actualURL,
                relativePath: relativePath
            )
            treeHasher.update(data: Data(relativePath.utf8))
            treeHasher.update(data: Data([0]))
            treeHasher.update(data: Data(digest))
            treeHasher.update(data: Data([0]))
            byteCount += expectedSize
        }

        return Comparison(
            fileCount: expectedFiles.count,
            byteCount: byteCount,
            sha256: hex(treeHasher.finalize())
        )
    }

    private static func describeDifference(missing: [String], extra: [String]) -> String {
        var parts: [String] = []
        if !missing.isEmpty {
            parts.append(" missing=\(missing.prefix(5).joined(separator: ","))")
        }
        if !extra.isEmpty {
            parts.append(" extra=\(extra.prefix(5).joined(separator: ","))")
        }
        if missing.count > 5 || extra.count > 5 {
            parts.append(" ...")
        }
        return parts.joined()
    }

    private static func relativeRegularFiles(in root: URL) throws -> [String: URL] {
        let rootPath = root.standardizedFileURL.path
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: []
        ) else {
            throw MLXFastError.missingFile("directory not found at \(root.path)")
        }

        var files: [String: URL] = [:]
        for case let url as URL in enumerator {
            let standardized = url.standardizedFileURL
            let path = standardized.path
            guard path.hasPrefix(rootPath + "/") else {
                throw MLXFastError.invalidInput("path escaped transform verification root: \(path)")
            }
            let relativePath = String(path.dropFirst(rootPath.count + 1))
            if ignoredRelativePath(relativePath) {
                continue
            }

            let values = try standardized.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput(
                    "transform verification rejects symlink \(relativePath)"
                )
            }
            if values.isDirectory == true {
                continue
            }
            guard values.isRegularFile == true else {
                throw MLXFastError.invalidInput(
                    "transform verification rejects non-regular file \(relativePath)"
                )
            }
            guard !relativePath.contains("\n") else {
                throw MLXFastError.invalidInput(
                    "transform verification rejects newline in path \(relativePath)"
                )
            }
            files[relativePath] = standardized
        }
        return files
    }

    private static func ignoredRelativePath(_ path: String) -> Bool {
        path == ".benchmark-source.sha256" || path == ".gitkeep"
    }

    private static func byteCountOfFile(_ url: URL) throws -> Int {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return try fileSizeByteCount(from: attributes, path: url.path)
    }

    private static func compareFileBytes(
        expected: URL,
        actual: URL,
        relativePath: String
    ) throws -> SHA256.Digest {
        let expectedHandle = try FileHandle(forReadingFrom: expected)
        let actualHandle = try FileHandle(forReadingFrom: actual)
        defer {
            try? expectedHandle.close()
            try? actualHandle.close()
        }

        var hasher = SHA256()
        let chunkSize = 8 * 1024 * 1024
        while true {
            let expectedData = expectedHandle.readData(ofLength: chunkSize)
            let actualData = actualHandle.readData(ofLength: chunkSize)
            guard expectedData == actualData else {
                throw MLXFastError.invalidInput(
                    "transform verification byte mismatch for \(relativePath)"
                )
            }
            if expectedData.isEmpty {
                return hasher.finalize()
            }
            hasher.update(data: expectedData)
        }
    }

    private static func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
