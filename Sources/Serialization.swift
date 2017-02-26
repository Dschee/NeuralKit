//
//  Serialization.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 26.02.17.
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

import Foundation
import SwiftyJSON

public protocol Serializable
{
	init?(_ json: JSON)
	func serialize() -> JSON
}

extension Matrix: Serializable
{
	public init?(_ json: JSON)
	{
		guard
			let width = json["width"].int,
			let height = json["height"].int,
			let values = json["values"].array?.flatMap({$0.float})
		else
		{
			return nil
		}
		self.init(values: values, width: width, height: height)
	}
	
	public func serialize() -> JSON
	{
		return [
			"width" : self.width,
			"height" : self.height,
			"values" : self.values
		]
	}
}
