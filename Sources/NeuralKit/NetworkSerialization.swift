//
//  NetworkSerialization.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 03.03.17.
//	Copyright (c) 2017 Palle Klewitz
//
//	Permission is hereby granted, free of charge, to any person obtaining a copy
//	of this software and associated documentation files (the "Software"), to deal
//	in the Software without restriction, including without limitation the rights
//	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//	copies of the Software, and to permit persons to whom the Software is
//	furnished to do so, subject to the following conditions:
//
//	The above copyright notice and this permission notice shall be included in all
//	copies or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//	SOFTWARE.

//MARK: Matrices

import Serialization
import MatrixVector

//MARK: Activation


extension Activation: Serializable
{
	
	public func serialized() -> Any
	{
		switch self
		{
		case .linear:
			return "linear"
			
		case .relu:
			return "relu"
			
		case .sigmoid:
			return "sigmoid"
			
		case .tanh:
			return "tanh"
			
		case .softmax:
			return "softmax"
		}
	}
	
	
	public init(json: Any) throws
	{
		guard let activationName = json as? String else
		{
			throw DecodingError.invalidType(expected: "String", actual: json)
		}
		switch activationName
		{
		case "linear":
			self = .linear
			
		case "relu":
			self = .relu
			
		case "sigmoid":
			self = .sigmoid
			
		case "tanh":
			self = .tanh
			
		case "softmax":
			self = .softmax
			
		default:
			throw DecodingError.invalidValue(expected: "linear|relu|sigmoid|tanh", actual: activationName)
		}
	}
	
}


//MARK: Layers


extension FullyConnectedLayer: Serializable
{
	
	public func serialized() -> Any
	{
		return [
			"weights": _weights.serialized()
		]
	}
	
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard
			let weights = try data["weights"].flatMap(Matrix.init(json:))
		else
		{
			throw DecodingError.missingKey(key: "weights", data: data)
		}
		
		self.init(weights: weights)
	}
	
}


extension ConvolutionLayer: Serializable
{
	public func serialized() -> Any
	{
		return [
			"input_size": [
				"width": inputSize.width,
				"height": inputSize.height,
				"depth": inputSize.depth
			],
			"kernels": zip(kernels, bias).map
			{[
				"kernel": $0.serialized(),
				"bias": $1
			]},
			"horizontal_stride": horizontalStride,
			"vertical_stride": verticalStride,
			"horizontal_inset": horizontalInset,
			"vertical_inset": verticalInset
		]
	}
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard
			let inputSize = data["input_size"] as? [String: Int],
			let kernelData = data["kernels"] as? [String: Any]
		else
		{
			throw DecodingError.missingKey(key: "input_size|kernels", data: data)
		}
		
		guard
			let inputWidth = inputSize["width"],
			let inputHeight = inputSize["height"],
			let inputDepth = inputSize["depth"]
		else
		{
			throw DecodingError.missingKey(key: "input_size.(width|height|depth)", data: data)
		}
		
		guard
			let horizontalStride = data["horizontal_stride"] as? Int,
			let verticalStride = data["vertical_stride"] as? Int,
			let horizontalInset = data["horizontal_inset"] as? Int,
			let verticalInset = data["vertical_inset"] as? Int
		else
		{
			throw DecodingError.missingKey(key: "horizontal_stride|vertical_stride|horizontal_inset|vertical_inset", data: data)
		}
		
		let bias = kernelData
			.filter{$0.key == "bias"}
			.flatMap{$0.value as? Float}
		
		let kernels = try kernelData
			.filter{$0.key == "kernel"}
			.flatMap{$0.value}
			.flatMap(Matrix3.init(json: ))
		
		self.init(
			inputSize: (width: inputWidth, height: inputHeight, depth: inputDepth),
			kernels: kernels,
			bias: bias,
			horizontalStride: horizontalStride,
			verticalStride: verticalStride,
			horizontalInset: horizontalInset,
			verticalInset: verticalInset
		)
	}
}


extension PoolingLayer: Serializable
{
	public func serialized() -> Any
	{
		return [
			"input_size": [
				"width": inputSize.width,
				"height": inputSize.height,
				"depth": inputSize.depth
			],
			"output_size": [
				"width": outputSize.width,
				"height": outputSize.height,
				"depth": outputSize.depth
			]
		]
	}
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard
			let inputSize = data["input_size"] as? [String: Int],
			let outputSize = data["output_size"] as? [String: Int]
		else
		{
			throw DecodingError.missingKey(key: "input_size|output_size", data: data)
		}
		
		guard
			let inputWidth = inputSize["width"],
			let inputHeight = inputSize["height"],
			let inputDepth = inputSize["depth"],
			let outputWidth = outputSize["width"],
			let outputHeight = outputSize["height"],
			let outputDepth = outputSize["depth"]
		else
		{
			throw DecodingError.missingKey(key: "(input_size|output_size).(width|height|depth)", data: data)
		}
		
		self.init(
			inputSize: (width: inputWidth, height: inputHeight, depth: inputDepth),
			outputSize: (width: outputWidth, height: outputHeight, depth: outputDepth)
		)
	}
}


extension ReshapingLayer: Serializable
{
	public func serialized() -> Any
	{
		return [
			"input_size": [
				"width": inputSize.width,
				"height": inputSize.height,
				"depth": inputSize.depth
			],
			"output_size": [
				"width": outputSize.width,
				"height": outputSize.height,
				"depth": outputSize.depth
			]
		]
	}
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard
			let inputSize = data["input_size"] as? [String: Int],
			let outputSize = data["output_size"] as? [String: Int]
			else
		{
			throw DecodingError.missingKey(key: "input_size|output_size", data: data)
		}
		
		guard
			let inputWidth = inputSize["width"],
			let inputHeight = inputSize["height"],
			let inputDepth = inputSize["depth"],
			let outputWidth = outputSize["width"],
			let outputHeight = outputSize["height"],
			let outputDepth = outputSize["depth"]
			else
		{
			throw DecodingError.missingKey(key: "(input_size|output_size).(width|height|depth)", data: data)
		}
		
		self.init(
			inputSize: (width: inputWidth, height: inputHeight, depth: inputDepth),
			outputSize: (width: outputWidth, height: outputHeight, depth: outputDepth)
		)
	}
}

extension NonlinearityLayer: Serializable
{
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard let activation = try data["activation"].flatMap(Activation.init(json: )) else
		{
			throw DecodingError.missingKey(key: "activation", data: data)
		}
		
		guard let inputSize = data["input_size"] as? [String: Int] else
		{
			throw DecodingError.missingKey(key: "input_size", data: data)
		}
		
		guard
			let inputWidth = inputSize["width"],
			let inputHeight = inputSize["height"],
			let inputDepth = inputSize["depth"]
			else
		{
			throw DecodingError.missingKey(key: "input_size.(width|height|depth)", data: data)
		}
		
		self.init(inputSize: (width: inputWidth, height: inputHeight, depth: inputDepth), activation: activation)
	}
	
	public func serialized() -> Any
	{
		return [
			"activation": self.activation.serialized(),
			"input_size": [
				"width": inputSize.width,
				"height": inputSize.height,
				"depth": inputSize.depth
			]
		]
	}
}


/// Utility for encoding neural layers.
/// 
/// Encodes the type of a layer next to it, so it can be correctly restored 
/// during deserialization
public struct NeuralLayerEncoder
{
	
	/// Do not use this.
	private init(){}
	
	
	/// Registered layer types which the encoder can decode.
	fileprivate static var layerTypes:[String: (NeuralLayer & Serializable).Type] = [:]
	
	
	/// Registers the default layer types (fully connected, convolutional, etc...)
	private static func registerDefaults()
	{
		layerTypes["\(FullyConnectedLayer.self)"] = FullyConnectedLayer.self
		layerTypes["\(PoolingLayer.self)"] = PoolingLayer.self
		layerTypes["\(ReshapingLayer.self)"] = ReshapingLayer.self
		//TODO: TODO: Other layer types
	}
	
	
	/// Registers a custom layer type
	///
	/// - Parameter layerType: Type of the layer which the encoder should be able to decode.
	public static func registerLayerType(_ layerType: (NeuralLayer & Serializable).Type)
	{
		if layerTypes.isEmpty
		{
			registerDefaults()
		}
		
		layerTypes["\(layerType)"] = layerType
	}
	
	
	/// Serializes a layer and also stores its type for deserialization.
	///
	/// If the layer is not included in the set of default layers
	/// (fully connected layer, convolutional layer, pooling layer)
	/// it must be registered before deserialization using the
	/// `NeuralLayerEncoder.registerLayerType(:)` method.
	///
	/// - Parameter layer: Layer which should be serialized
	/// - Returns: Serialized layer
	public static func serialize(_ layer: NeuralLayer & Serializable) -> Any
	{
		return [
			"data": layer.serialized(),
			"type": "\(type(of: layer))"
		]
	}
	
	
	/// Deserializes a layer which was encoded using the encode function.
	///
	/// If the layer is not included in the set of default layers
	/// (fully connected layer, convolutional layer, pooling layer)
	/// it must be registered before deserialization using the
	/// `NeuralLayerEncoder.registerLayerType(:)` method.
	///
	/// - Parameter json: Serialized representation of the layer
	/// - Returns: Deserialized layer
	/// - Throws: An error if the data is in the incorrect format or is
	/// missing required values.
	public static func deserialize(_ json: Any) throws -> NeuralLayer
	{
		if layerTypes.isEmpty
		{
			registerDefaults()
		}
		
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard let typeName = data["type"] as? String else
		{
			throw DecodingError.missingKey(key: "type", data: data)
		}
		guard let LayerType = layerTypes[typeName] else
		{
			throw DecodingError.invalidValue(expected: layerTypes.keys.joined(separator: "|"), actual: typeName)
		}
		guard let layerData = data["data"] else
		{
			throw DecodingError.missingKey(key: "data", data: data)
		}
		return try LayerType.init(json: layerData)
	}
}


//MARK: Network


extension FeedForwardNeuralNetwork: Serializable
{
	
	public func serialized() -> Any
	{
		return [
			"layers": layers.flatMap{$0 as? (NeuralLayer & Serializable)}.map(NeuralLayerEncoder.serialize),
			"output_activation": outputLayer.serialized()
		]
	}
	
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard let layers = try (data["layers"] as? [Any])?.map(NeuralLayerEncoder.deserialize) else
		{
			throw DecodingError.missingKey(key: "layers", data: data)
		}
		guard let outputLayer = try data["output_activation"].flatMap(NeuralLayerEncoder.deserialize) else
		{
			throw DecodingError.missingKey(key: "output_activation", data: data)
		}
		guard let network = FeedForwardNeuralNetwork(layers: layers as! [BidirectionalLayer], outputLayer: outputLayer as! OutputLayer) else
		{
			throw DecodingError.invalidValue(expected: "Layer must have matching input size to precedent layer.", actual: data)
		}
		self = network
	}
	
}
