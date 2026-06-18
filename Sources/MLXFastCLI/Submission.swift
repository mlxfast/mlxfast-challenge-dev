import Foundation
import MLXFastCore

struct ChallengeContract: Decodable, Equatable {
    let schemaVersion: Int
    let name: String
    let editablePaths: [String]
}

struct SubmissionArchiveReport: Codable, Equatable {
    let contractPath: String
    let archivePath: String
    let editablePaths: [String]
    let fileCount: Int
    let byteCount: Int
}

enum SubmissionSupport {
    static func ensureWorkspace(contractPath: String) throws -> ChallengeContract {
        let contract = try loadContract(at: contractPath)
        _ = try editableFiles(from: contract, contractPath: contractPath)
        return contract
    }

    static func packageEditablePaths(
        contractPath: String,
        outputPath: String
    ) throws -> SubmissionArchiveReport {
        let contract = try loadContract(at: contractPath)
        let files = try editableFiles(from: contract, contractPath: contractPath)
        guard !files.isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json editablePaths did not select any files")
        }

        let archiveURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        if let parent = archiveURL.deletingLastPathComponentIfPresent() {
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }
        if FileManager.default.fileExists(atPath: archiveURL.path) {
            try FileManager.default.removeItem(at: archiveURL)
        }

        let contractRoot = URL(fileURLWithPath: contractPath).standardizedFileURL
            .deletingLastPathComponent()
        try runZip(
            relativeFiles: files.map(\.relativePath),
            archivePath: archiveURL.path,
            workingDirectory: contractRoot
        )
        let byteCount = try files.reduce(0) { partial, file in
            partial + (try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: file.absoluteURL.path),
                path: file.absoluteURL.path
            ))
        }

        return SubmissionArchiveReport(
            contractPath: URL(fileURLWithPath: contractPath).standardizedFileURL.path,
            archivePath: archiveURL.path,
            editablePaths: contract.editablePaths,
            fileCount: files.count,
            byteCount: byteCount
        )
    }

    static func storeCredentials(apiKey: String) throws -> String {
        let trimmed = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MLXFastError.invalidInput("login requires a non-empty API key")
        }
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config", isDirectory: true)
            .appendingPathComponent("mlxfast", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("credentials.json")
        let object: [String: Any] = [
            "api_key": trimmed,
            "stored_at": Date().timeIntervalSince1970,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try data.write(to: path, options: [.atomic])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
        return path.path
    }

    private struct EditableFile: Equatable {
        let relativePath: String
        let absoluteURL: URL
    }

    private static func loadContract(at path: String) throws -> ChallengeContract {
        try requireFile(path, description: "benchmark contract")
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let contract = try JSONDecoder().decode(ChallengeContract.self, from: data)
        guard contract.schemaVersion == 1 else {
            throw MLXFastError.invalidInput("benchmark.json schemaVersion must be 1")
        }
        guard !contract.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json name must not be empty")
        }
        guard !contract.editablePaths.isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json editablePaths must not be empty")
        }
        for path in contract.editablePaths {
            try validateRelativeContractPath(path, field: "editablePaths")
        }
        return contract
    }

    private static func editableFiles(
        from contract: ChallengeContract,
        contractPath: String
    ) throws -> [EditableFile] {
        let contractURL = URL(fileURLWithPath: contractPath).standardizedFileURL
        let root = contractURL.deletingLastPathComponent()
        var files: [EditableFile] = []
        for editablePath in contract.editablePaths {
            let rootURL = root.appendingPathComponent(editablePath).standardizedFileURL
            try ensureInside(rootURL, root: root, relativePath: editablePath)
            guard FileManager.default.fileExists(atPath: rootURL.path) else {
                throw MLXFastError.missingFile("editable path \(editablePath) not found")
            }
            let values = try rootURL.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput("editable path \(editablePath) must not be a symlink")
            }
            if values.isRegularFile == true {
                files.append(EditableFile(relativePath: editablePath, absoluteURL: rootURL))
                continue
            }
            guard values.isDirectory == true else {
                throw MLXFastError.invalidInput("editable path \(editablePath) must be a file or directory")
            }
            files.append(contentsOf: try regularFilesUnder(rootURL, root: root))
        }
        return files.sorted { $0.relativePath < $1.relativePath }
    }

    private static func regularFilesUnder(_ directory: URL, root: URL) throws -> [EditableFile] {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: []
        ) else {
            throw MLXFastError.missingFile("editable directory not found at \(directory.path)")
        }

        var files: [EditableFile] = []
        for case let url as URL in enumerator {
            let standardized = url.standardizedFileURL
            let relativePath = try relativePathForURL(standardized, root: root)
            let values = try standardized.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput(
                    "editable path \(relativePath) must not be a symlink"
                )
            }
            if values.isDirectory == true {
                continue
            }
            guard values.isRegularFile == true else {
                throw MLXFastError.invalidInput(
                    "editable path \(relativePath) must be a regular file"
                )
            }
            files.append(EditableFile(relativePath: relativePath, absoluteURL: standardized))
        }
        return files
    }

    private static func validateRelativeContractPath(_ path: String, field: String) throws {
        guard !path.isEmpty else {
            throw MLXFastError.invalidInput("benchmark.json \(field) contains an empty path")
        }
        guard !path.hasPrefix("/") else {
            throw MLXFastError.invalidInput("benchmark.json \(field) path \(path) must be relative")
        }
        guard !path.contains("\0"), !path.contains("\n") else {
            throw MLXFastError.invalidInput("benchmark.json \(field) path \(path) contains invalid characters")
        }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.contains(""), !components.contains("."), !components.contains("..") else {
            throw MLXFastError.invalidInput("benchmark.json \(field) path \(path) is not normalized")
        }
    }

    private static func relativePathForURL(_ url: URL, root: URL) throws -> String {
        let rootPath = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath + "/") else {
            throw MLXFastError.invalidInput("editable path escaped repository root: \(path)")
        }
        let relativePath = String(path.dropFirst(rootPath.count + 1))
        try validateRelativeContractPath(relativePath, field: "editable file")
        return relativePath
    }

    private static func ensureInside(_ url: URL, root: URL, relativePath: String) throws {
        let resolved = url.standardizedFileURL.path
        let rootPath = root.standardizedFileURL.path
        guard resolved == rootPath || resolved.hasPrefix(rootPath + "/") else {
            throw MLXFastError.invalidInput(
                "benchmark.json editable path \(relativePath) escapes repository root"
            )
        }
    }

    private static func runZip(
        relativeFiles: [String],
        archivePath: String,
        workingDirectory: URL
    ) throws {
        let candidates = ["/usr/bin/zip", "/opt/homebrew/bin/zip", "/usr/local/bin/zip"]
        guard let zip = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw MLXFastError.missingFile("zip executable not found")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: zip)
        process.arguments = ["-q", "-X", archivePath, "-@"]
        process.currentDirectoryURL = workingDirectory
        let input = Pipe()
        let errorPipe = Pipe()
        process.standardInput = input
        process.standardError = errorPipe

        try process.run()
        let list = relativeFiles.joined(separator: "\n") + "\n"
        input.fileHandleForWriting.write(Data(list.utf8))
        try input.fileHandleForWriting.close()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let error = String(
                data: errorPipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
            throw MLXFastError.invalidInput("zip failed with status \(process.terminationStatus): \(error)")
        }
    }
}

private extension URL {
    func deletingLastPathComponentIfPresent() -> URL? {
        let parent = deletingLastPathComponent()
        return parent.path.isEmpty ? nil : parent
    }
}
