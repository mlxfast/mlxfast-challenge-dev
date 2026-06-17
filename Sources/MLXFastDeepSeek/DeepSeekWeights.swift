import Foundation
import MLX
import MLXFastCore

public enum DeepSeekWeightNames {
    public static func model(_ suffix: String) -> [String] {
        [
            "model.\(suffix)",
            "language_model.model.\(suffix)",
        ]
    }

    public static func layer(_ layerIndex: Int, _ suffix: String) -> [String] {
        model("layers.\(layerIndex).\(suffix)")
    }

    public static let embedTokens = model("embed_tokens.weight")
    public static let finalNorm = model("norm.weight")
    public static let hcHeadFn = model("hc_head.fn")
    public static let hcHeadBase = model("hc_head.base")
    public static let hcHeadScale = model("hc_head.scale")

    public static let lmHead = [
        "lm_head.weight",
        "language_model.lm_head.weight",
    ]

    public static func attention(_ layerIndex: Int, _ suffix: String) -> [String] {
        layer(layerIndex, "attn.\(suffix)")
    }

    public static func feedForward(_ layerIndex: Int, _ suffix: String) -> [String] {
        layer(layerIndex, "ffn.\(suffix)")
    }

    public static func routedExpert(
        layerIndex: Int,
        expertIndex: Int,
        projection: DeepSeekExpertProjection
    ) -> [String] {
        let sanitized = projection.rawValue
        var candidates = [
            "ffn.switch_mlp.\(sanitized).weight",
            "ffn.switch_mlp.\(expertIndex).\(sanitized).weight",
            "ffn.experts.\(expertIndex).\(sanitized).weight",
        ].flatMap { layer(layerIndex, $0) }

        if let legacy = projection.legacyName {
            candidates += layer(layerIndex, "ffn.experts.\(expertIndex).\(legacy).weight")
        }
        return candidates
    }

    public static func attentionNorm(_ layerIndex: Int) -> [String] {
        layer(layerIndex, "attn_norm.weight")
    }

    public static func feedForwardNorm(_ layerIndex: Int) -> [String] {
        layer(layerIndex, "ffn_norm.weight")
    }

    public static func hyperConnection(
        layerIndex: Int,
        block: DeepSeekHyperConnectionBlock,
        component: DeepSeekHyperConnectionComponent
    ) -> [String] {
        layer(layerIndex, "\(block.rawValue).\(component.rawValue)")
    }
}

public enum DeepSeekHyperConnectionBlock: String, Equatable {
    case attention = "attn_hc"
    case feedForward = "ffn_hc"
}

public enum DeepSeekHyperConnectionComponent: String, Equatable {
    case fn
    case base
    case scale
}

public enum DeepSeekExpertProjection: String, Equatable, Hashable {
    case gate = "gate_proj"
    case up = "up_proj"
    case down = "down_proj"

    public var legacyName: String? {
        switch self {
        case .gate:
            return "w1"
        case .down:
            return "w2"
        case .up:
            return "w3"
        }
    }
}

public struct DeepSeekWeightLoader {
    public let denseStore: DenseTensorStore
    public let expertBank: ExpertSlotBank
    public let expertStreamingConfig: ExpertStreamingConfig
    public let expertStreamingMetrics: ExpertStreamingMetrics?
    private let bridge: MLXArrayTensorBridge

    public init(
        weightsPath: String,
        expertStreamingConfig: ExpertStreamingConfig = .fromEnvironment(),
        expertStreamingMetrics: ExpertStreamingMetrics? = nil,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) throws {
        let metrics = expertStreamingMetrics
            ?? (expertStreamingConfig.recordsMetrics ? ExpertStreamingMetrics() : nil)
        self.denseStore = try DenseTensorStore(weightsPath: weightsPath)
        self.expertStreamingConfig = expertStreamingConfig
        self.expertStreamingMetrics = metrics
        self.expertBank = try ExpertSlotBank(
            manifestPath: "\(weightsPath)/experts/manifest.json",
            capacity: expertStreamingConfig.tensorCacheCapacity,
            metrics: metrics
        )
        self.bridge = bridge
    }

    public init(
        denseStore: DenseTensorStore,
        expertBank: ExpertSlotBank,
        expertStreamingConfig: ExpertStreamingConfig = ExpertStreamingConfig(),
        expertStreamingMetrics: ExpertStreamingMetrics? = nil,
        bridge: MLXArrayTensorBridge = MLXArrayTensorBridge()
    ) {
        self.denseStore = denseStore
        self.expertBank = expertBank
        self.expertStreamingConfig = expertStreamingConfig
        self.expertStreamingMetrics = expertStreamingMetrics ?? expertBank.metrics
        self.bridge = bridge
    }

    public func resolveDenseName(_ candidates: [String]) throws -> String {
        for candidate in candidates where denseStore.record(named: candidate) != nil {
            return candidate
        }
        throw MLXFastError.invalidInput(
            "dense tensor not found; tried \(candidates.joined(separator: ", "))"
        )
    }

    public func materializedDenseTensor(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MaterializedTensor {
        let name = try resolveDenseName(candidates)
        let tensor = try denseStore.materializedTensor(named: name)
        try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: name)
        return tensor
    }

    public func denseArray(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MLXArray {
        try bridge.makeArray(
            from: materializedDenseTensor(
                candidates: candidates,
                expectedShape: expectedShape
            )
        )
    }

    public func optionalDenseArray(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MLXArray? {
        for candidate in candidates where denseStore.record(named: candidate) != nil {
            return try denseArray(candidates: [candidate], expectedShape: expectedShape)
        }
        return nil
    }

    public func denseLinearWeight(
        candidates: [String],
        expectedShape: [Int]
    ) throws -> DeepSeekLinearWeight {
        let name = try resolveDenseName(candidates)
        return try linearWeight(
            baseName: name,
            expectedShape: expectedShape,
            tensor: denseStore.materializedTensor(named: name),
            companionTensor: { companionName, _ in
                guard denseStore.record(named: companionName) != nil else {
                    return nil
                }
                return try denseStore.materializedTensor(named: companionName)
            }
        )
    }

    public func materializedExpertTensor(
        named name: String,
        expectedShape: [Int]? = nil
    ) throws -> MaterializedTensor {
        let tensor = try expertBank.materializedTensor(named: name)
        try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: name)
        return tensor
    }

    public func expertArray(named name: String, expectedShape: [Int]? = nil) throws -> MLXArray {
        try bridge.makeArray(
            from: materializedExpertTensor(
                named: name,
                expectedShape: expectedShape
            )
        )
    }

    public func resolveExpertName(_ candidates: [String]) throws -> String {
        for candidate in candidates where expertBank.record(named: candidate) != nil {
            return candidate
        }
        throw MLXFastError.invalidInput(
            "expert tensor not found; tried \(candidates.joined(separator: ", "))"
        )
    }

    public func expertArray(
        candidates: [String],
        expectedShape: [Int]? = nil
    ) throws -> MLXArray {
        try expertArray(
            named: resolveExpertName(candidates),
            expectedShape: expectedShape
        )
    }

    public func expertLinearWeight(
        candidates: [String],
        expectedShape: [Int],
        expertIndex: Int
    ) throws -> DeepSeekLinearWeight {
        for candidate in candidates {
            guard let record = expertBank.record(named: candidate) else {
                continue
            }
            let isStacked = record.shape.count == expectedShape.count + 1
                && record.shape.first.map { expertIndex < $0 } == true
            let tensor = try isStacked
                ? expertBank.materializedTensor(named: candidate, firstAxisIndex: expertIndex)
                : expertBank.materializedTensor(named: candidate)
            return try linearWeight(
                baseName: candidate,
                expectedShape: expectedShape,
                tensor: tensor,
                companionTensor: { companionName, shouldSlice in
                    guard expertBank.record(named: companionName) != nil else {
                        return nil
                    }
                    return try shouldSlice
                        ? expertBank.materializedTensor(named: companionName, firstAxisIndex: expertIndex)
                        : expertBank.materializedTensor(named: companionName)
                },
                shouldSliceCompanions: isStacked
            )
        }
        throw MLXFastError.invalidInput(
            "expert tensor not found; tried \(candidates.joined(separator: ", "))"
        )
    }

    public func embedTokens(expectedShape: [Int]) throws -> DeepSeekLinearWeight {
        try denseLinearWeight(candidates: DeepSeekWeightNames.embedTokens, expectedShape: expectedShape)
    }

    public func lmHead(expectedShape: [Int]) throws -> DeepSeekLinearWeight {
        try denseLinearWeight(candidates: DeepSeekWeightNames.lmHead, expectedShape: expectedShape)
    }

    public func finalNorm(expectedShape: [Int]? = nil) throws -> MLXArray {
        try denseArray(candidates: DeepSeekWeightNames.finalNorm, expectedShape: expectedShape)
    }

    public func modelWeights(config: DeepSeekConfig) throws -> DeepSeekModelWeights {
        try DeepSeekModelWeights(
            embedTokens: embedTokens(expectedShape: [config.vocabSize, config.hiddenSize]),
            finalNorm: finalNorm(expectedShape: [config.hiddenSize]),
            headHyperConnection: headHyperConnectionWeights(config: config),
            lmHead: lmHead(expectedShape: [config.vocabSize, config.hiddenSize])
        )
    }

    public func headHyperConnectionWeights(config: DeepSeekConfig) throws -> DeepSeekHeadHyperConnectionWeights {
        try DeepSeekHeadHyperConnectionWeights(
            fn: denseArray(
                candidates: DeepSeekWeightNames.hcHeadFn,
                expectedShape: [config.hcMult, config.hcMult * config.hiddenSize]
            ),
            base: denseArray(
                candidates: DeepSeekWeightNames.hcHeadBase,
                expectedShape: [config.hcMult]
            ),
            scale: denseArray(
                candidates: DeepSeekWeightNames.hcHeadScale,
                expectedShape: [1]
            )
        )
    }

    public func localAttentionWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekLocalAttentionWeights {
        try localAttentionWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            qLoraRank: config.qLoraRank,
            outputLoraRank: config.outputLoraRank,
            spec: DeepSeekLocalAttentionSpec(config: config),
            attentionBias: config.attentionBias
        )
    }

    public func localAttentionWeights(
        layerIndex: Int,
        hiddenSize: Int,
        qLoraRank: Int,
        outputLoraRank: Int,
        spec: DeepSeekLocalAttentionSpec,
        attentionBias: Bool = false
    ) throws -> DeepSeekLocalAttentionWeights {
        let groupedInput = spec.numAttentionHeads * spec.headDim / spec.outputGroups
        return try DeepSeekLocalAttentionWeights(
            wqA: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wq_a.weight"),
                expectedShape: [qLoraRank, hiddenSize]
            ),
            qNorm: denseArray(
                candidates: DeepSeekWeightNames.attention(layerIndex, "q_norm.weight"),
                expectedShape: [qLoraRank]
            ),
            wqB: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wq_b.weight"),
                expectedShape: [spec.numAttentionHeads * spec.headDim, qLoraRank]
            ),
            wkv: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wkv.weight"),
                expectedShape: [spec.headDim, hiddenSize]
            ),
            kvNorm: denseArray(
                candidates: DeepSeekWeightNames.attention(layerIndex, "kv_norm.weight"),
                expectedShape: [spec.headDim]
            ),
            woA: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wo_a.weight"),
                expectedShape: [spec.outputGroups, outputLoraRank, groupedInput]
            ),
            woB: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "wo_b.weight"),
                expectedShape: [hiddenSize, spec.outputGroups * outputLoraRank]
            ),
            woBBias: attentionBias
                ? optionalDenseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "wo_b.bias"),
                    expectedShape: [hiddenSize]
                )
                : nil,
            attentionSink: optionalDenseArray(
                candidates: DeepSeekWeightNames.attention(layerIndex, "attn_sink"),
                expectedShape: [spec.numAttentionHeads]
            )
        )
    }

    public func compressedAttentionWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekCompressedAttentionWeights {
        let ratio = config.compressRatios[layerIndex]
        let outDim = config.headDim * (ratio == 4 ? 2 : 1)
        return try DeepSeekCompressedAttentionWeights(
            attention: localAttentionWeights(layerIndex: layerIndex, config: config),
            compressor: DeepSeekCompressorWeights(
                wkv: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.wkv.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                wgate: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.wgate.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                ape: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.ape"),
                    expectedShape: [ratio, outDim]
                ),
                norm: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "compressor.norm.weight"),
                    expectedShape: [config.headDim]
                )
            ),
            indexer: ratio == 4 ? indexerWeights(layerIndex: layerIndex, config: config) : nil
        )
    }

    public func indexerWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekIndexerWeights {
        let ratio = config.compressRatios[layerIndex]
        let outDim = config.indexHeadDim * (ratio == 4 ? 2 : 1)
        return try DeepSeekIndexerWeights(
            wqB: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.wq_b.weight"),
                expectedShape: [config.indexHeads * config.indexHeadDim, config.qLoraRank]
            ),
            weightsProj: denseLinearWeight(
                candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.weights_proj.weight"),
                expectedShape: [config.indexHeads, config.hiddenSize]
            ),
            compressor: DeepSeekCompressorWeights(
                wkv: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wkv.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                wgate: denseLinearWeight(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.wgate.weight"),
                    expectedShape: [outDim, config.hiddenSize]
                ),
                ape: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.ape"),
                    expectedShape: [ratio, outDim]
                ),
                norm: denseArray(
                    candidates: DeepSeekWeightNames.attention(layerIndex, "indexer.compressor.norm.weight"),
                    expectedShape: [config.indexHeadDim]
                )
            )
        )
    }

    public func blockWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekBlockWeights {
        try blockWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            spec: DeepSeekBlockSpec(config: config)
        )
    }

    public func blockWeights(
        layerIndex: Int,
        hiddenSize: Int,
        spec: DeepSeekBlockSpec
    ) throws -> DeepSeekBlockWeights {
        try DeepSeekBlockWeights(
            attentionNorm: denseArray(
                candidates: DeepSeekWeightNames.attentionNorm(layerIndex),
                expectedShape: [hiddenSize]
            ),
            feedForwardNorm: denseArray(
                candidates: DeepSeekWeightNames.feedForwardNorm(layerIndex),
                expectedShape: [hiddenSize]
            ),
            attentionHyperConnection: hyperConnectionWeights(
                layerIndex: layerIndex,
                block: .attention,
                hiddenSize: hiddenSize,
                spec: spec
            ),
            feedForwardHyperConnection: hyperConnectionWeights(
                layerIndex: layerIndex,
                block: .feedForward,
                hiddenSize: hiddenSize,
                spec: spec
            )
        )
    }

    public func sharedMLPWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekMLPWeights {
        try sharedMLPWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.sharedExperts
        )
    }

    public func moeWeights(
        layerIndex: Int,
        config: DeepSeekConfig
    ) throws -> DeepSeekMoEWeights {
        try moeWeights(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            routedExperts: config.routedExperts,
            vocabSize: config.vocabSize,
            expertsPerToken: config.expertsPerToken,
            sharedIntermediateSize: config.moeIntermediateSize * config.sharedExperts,
            isHashLayer: layerIndex < config.numHashLayers
        )
    }

    public func moeWeights(
        layerIndex: Int,
        hiddenSize: Int,
        routedExperts: Int,
        vocabSize: Int,
        expertsPerToken: Int,
        sharedIntermediateSize: Int,
        isHashLayer: Bool
    ) throws -> DeepSeekMoEWeights {
        try DeepSeekMoEWeights(
            gate: denseArray(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.weight"),
                expectedShape: [routedExperts, hiddenSize]
            ),
            correctionBias: isHashLayer
                ? nil
                : optionalDenseArray(
                    candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.e_score_correction_bias"),
                    expectedShape: [routedExperts]
                ),
            tokenToExpert: isHashLayer
                ? try optionalDenseArray(
                    candidates: DeepSeekWeightNames.feedForward(layerIndex, "gate.tid2eid"),
                    expectedShape: [vocabSize, expertsPerToken]
                )
                : nil,
            sharedExperts: sharedMLPWeights(
                layerIndex: layerIndex,
                hiddenSize: hiddenSize,
                intermediateSize: sharedIntermediateSize
            )
        )
    }

    public func sharedMLPWeights(
        layerIndex: Int,
        hiddenSize: Int,
        intermediateSize: Int
    ) throws -> DeepSeekMLPWeights {
        try DeepSeekMLPWeights(
            gate: denseLinearWeight(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.gate_proj.weight"),
                expectedShape: [intermediateSize, hiddenSize]
            ),
            up: denseLinearWeight(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.up_proj.weight"),
                expectedShape: [intermediateSize, hiddenSize]
            ),
            down: denseLinearWeight(
                candidates: DeepSeekWeightNames.feedForward(layerIndex, "shared_experts.down_proj.weight"),
                expectedShape: [hiddenSize, intermediateSize]
            )
        )
    }

    public func hyperConnectionWeights(
        layerIndex: Int,
        block: DeepSeekHyperConnectionBlock,
        hiddenSize: Int,
        spec: DeepSeekBlockSpec
    ) throws -> DeepSeekHyperConnectionWeights {
        let mix = (2 + spec.hcMult) * spec.hcMult
        return try DeepSeekHyperConnectionWeights(
            fn: denseArray(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .fn
                ),
                expectedShape: [mix, spec.hcMult * hiddenSize]
            ),
            base: denseArray(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .base
                ),
                expectedShape: [mix]
            ),
            scale: denseArray(
                candidates: DeepSeekWeightNames.hyperConnection(
                    layerIndex: layerIndex,
                    block: block,
                    component: .scale
                ),
                expectedShape: [3]
            )
        )
    }

    private func validateShape(
        _ actualShape: [Int],
        expectedShape: [Int]?,
        tensorName: String
    ) throws {
        guard let expectedShape else {
            return
        }
        guard actualShape == expectedShape else {
            throw MLXFastError.invalidInput(
                "tensor \(tensorName) shape \(actualShape) does not match expected shape \(expectedShape)"
            )
        }
    }

    private func linearWeight(
        baseName: String,
        expectedShape: [Int],
        tensor: MaterializedTensor,
        companionTensor: (_ companionName: String, _ shouldSlice: Bool) throws -> MaterializedTensor?,
        shouldSliceCompanions: Bool = false
    ) throws -> DeepSeekLinearWeight {
        let scalesName = companionName(for: baseName, suffix: "scales")
        guard tensor.dtype == .u32, let scalesTensor = try companionTensor(scalesName, shouldSliceCompanions) else {
            try validateShape(tensor.shape, expectedShape: expectedShape, tensorName: baseName)
            return DeepSeekLinearWeight(try bridge.makeArray(from: tensor))
        }

        let biasesTensor = try companionTensor(
            companionName(for: baseName, suffix: "biases"),
            shouldSliceCompanions
        )
        let expectedRows = expectedShape.dropLast().reduce(1, *)
        guard
            let expectedInput = expectedShape.last,
            let packedInput = tensor.shape.last,
            expectedInput > 0,
            packedInput > 0
        else {
            throw MLXFastError.invalidInput("linear tensor \(baseName) has invalid expected shape \(expectedShape)")
        }
        let actualRows = tensor.shape.dropLast().reduce(1, *)
        guard actualRows == expectedRows else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) has \(actualRows) output rows; expected \(expectedRows) from \(expectedShape)"
            )
        }
        let packedBits = packedInput * 32
        guard packedBits % expectedInput == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) packed input \(packedInput) is incompatible with logical input \(expectedInput)"
            )
        }
        let bits = packedBits / expectedInput
        guard [2, 4, 8].contains(bits) else {
            throw MLXFastError.invalidInput("quantized tensor \(baseName) inferred unsupported bits=\(bits)")
        }
        guard let scaleGroups = scalesTensor.shape.last, scaleGroups > 0, expectedInput % scaleGroups == 0 else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales shape \(scalesTensor.shape) is incompatible with logical input \(expectedInput)"
            )
        }
        let scaleRows = scalesTensor.shape.dropLast().reduce(1, *)
        guard scaleRows == expectedRows else {
            throw MLXFastError.invalidInput(
                "quantized tensor \(baseName) scales have \(scaleRows) rows; expected \(expectedRows)"
            )
        }
        if let biasesTensor {
            let biasRows = biasesTensor.shape.dropLast().reduce(1, *)
            guard biasRows == expectedRows, biasesTensor.shape.last == scaleGroups else {
                throw MLXFastError.invalidInput(
                    "quantized tensor \(baseName) biases shape \(biasesTensor.shape) does not match scales shape \(scalesTensor.shape)"
                )
            }
        }

        let mode: QuantizationMode = biasesTensor == nil && scalesTensor.dtype == .u8 ? .mxfp4 : .affine
        let weightArray = try bridge.makeArray(from: tensor).reshaped([expectedRows, packedInput])
        let scalesArray = try bridge.makeArray(from: scalesTensor).reshaped([expectedRows, scaleGroups])
        let biasesArray = try biasesTensor.map { try bridge.makeArray(from: $0).reshaped([expectedRows, scaleGroups]) }
        return DeepSeekLinearWeight(
            weight: weightArray,
            scales: scalesArray,
            biases: biasesArray,
            logicalShape: expectedShape,
            groupSize: expectedInput / scaleGroups,
            bits: bits,
            mode: mode
        )
    }

    private func companionName(for weightName: String, suffix: String) -> String {
        if weightName.hasSuffix(".weight") {
            return String(weightName.dropLast(".weight".count)) + ".\(suffix)"
        }
        return "\(weightName).\(suffix)"
    }
}
