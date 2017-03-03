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

extension Matrix: Serializable
{
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard
			let width = data["width"] as? Int,
			let height = data["height"] as? Int,
			let values = data["values"] as? [Float]
		else
		{
			throw DecodingError.missingKey(key: "width|height|values", data: json)
		}
		self.init(values: values, width: width, height: height)
	}
	
	
	public func serialized() -> Any
	{
		return [
			"width" : self.width,
			"height" : self.height,
			"values" : self.values
		]
	}
	
}


extension Matrix3: Serializable
{
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard
			let width = data["width"] as? Int,
			let height = data["height"] as? Int,
			let depth = data["depth"] as? Int,
			let values = data["values"] as? [Float]
		else
		{
			throw DecodingError.missingKey(key: "width|height|values", data: json)
		}
		self.init(values: values, width: width, height: height, depth: depth)
	}
	
	
	public func serialized() -> Any
	{
		return [
			"width" : self.width,
			"height" : self.height,
			"depth" : self.depth,
			"values" : self.values
		]
	}
	
}


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
			"activation": activationFunction.serialized(),
			"weights": weights.serialized()
		]
	}
	
	
	public init(json: Any) throws
	{
		guard let data = json as? [String: Any] else
		{
			throw DecodingError.invalidType(expected: "[String: Any]", actual: json)
		}
		guard
			let activation = try data["activation"].flatMap(Activation.init(json:)),
			let weights = try data["weights"].flatMap(Matrix.init(json:))
		else
		{
			throw DecodingError.missingKey(key: "activation|weights", data: data)
		}
		
		self.activationFunction = activation
		self.weights = weights
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


extension NeuralNetwork: Serializable
{
	
	public func serialized() -> Any
	{
		return [
			"layers": layers.flatMap{$0 as? (NeuralLayer & Serializable)}.map(NeuralLayerEncoder.serialize),
			"output_activation": outputActivationFunction.serialized()
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
		guard let outputActivation = try data["output_activation"].flatMap(Activation.init(json:)) else
		{
			throw DecodingError.missingKey(key: "output_activation", data: data)
		}
		self.init(layers: layers, outputActivation: outputActivation)
	}
	
}
