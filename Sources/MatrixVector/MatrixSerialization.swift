//
//  MatrixSerialization.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.04.17.
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
