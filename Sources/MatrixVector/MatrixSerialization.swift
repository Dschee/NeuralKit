//
//  MatrixSerialization.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.04.17.
//
//

import Foundation
import Serialization

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
