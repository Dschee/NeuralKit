//
//  Samples.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 19.02.17.
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

public protocol Sample
{
	var values: Matrix3 { get }
}

public struct InputSample: Sample
{
	public let values: Matrix3
}

public struct TrainingSample: Sample
{
	public let values: Matrix3
	public let expected: Matrix3
}

public extension TrainingSample
{
	public init(values: Matrix3, outputCount: Int, targetIndex: Int)
	{
		self.values = values
		self.expected = TrainingSample.encodeOneHot(count: outputCount, target: targetIndex)
	}
	
	public init(input: Matrix3, expected: Matrix3)
	{
		self.values = input
		self.expected = expected
	}
}

public extension Sample
{
	public static func normalize(samples: [InputSample]) -> ([InputSample], scale: Float, offset: Float)
	{
		guard
			let min = samples.flatMap({$0.values.values.min()}).min(),
			let max = samples.flatMap({$0.values.values.max()}).max()
		else
		{
			return (samples, scale: 1, offset: 0)
		}
		
		let scale = 1 / (max - min)
		
		return (
			Self.normalize(samples: samples, scale: scale, offset: -min),
			scale: scale,
			offset: -min
		)
	}
	
	public static func normalize(samples: [InputSample], scale: Float, offset: Float) -> [InputSample]
	{
		return samples
			.map { $0.values.values }
			.map { ($0 &+ offset) &* scale }
			// Force unwrapping because if no first element exists, this code would not be executed anyway:
			.map { Matrix3(values: $0, width: samples.first!.values.width, height: samples.first!.values.height, depth: samples.first!.values.depth) }
			.map { InputSample(values: $0) }
	}
	
	public static func denormalize(samples: [InputSample], scale: Float, offset: Float) -> [InputSample]
	{
		return Self.normalize(samples: samples, scale: 1 / scale, offset: -offset)
	}
	
	public static func encodeOneHot(count: Int, target: Int) -> Matrix3
	{
		precondition(count > target, "Target index greater than output length")
		var oneHotVector = [Float](repeating: 0.0, count: count)
		oneHotVector[target] = 1.0
		return Matrix3(values: oneHotVector, width: 1, height: 1, depth: count)
	}
}

