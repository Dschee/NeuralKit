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


/// A sample which can be forwarded through a neural network.
public protocol Sample
{
	
	/// Values of the sample in a three dimensional matrix
	var values: Matrix3 { get }
	
}


/// An input sample which can be forwarded through a neural network
public struct InputSample: Sample
{
	
	/// Values of the sample in a three dimensional matrix
	public let values: Matrix3
	
}


/// A training sample which also provides an expected output
public struct TrainingSample: Sample
{
	
	/// Values of the sample in a three dimensional matrix
	public let values: Matrix3
	
	
	/// Expected output values towards which a network can be trained
	public let expected: Matrix3
	
}


// Initializer extensions
public extension TrainingSample
{
	
	/// Initializes a training sample from an input and an expected output 
	/// which is a one-hot-vector.
	///
	/// A one-hot-vector contains zeros at all indices except one hot index.
	///
	/// The output will be a matrix with a width and height of one and a depth of `outputCount`.
	///
	/// - Parameters:
	///   - values: Input values
	///   - outputCount: Number of output values of a network which can be trained with this sample
	///   - targetIndex: Hot index which will be set to one.
	public init(values: Matrix3, outputCount: Int, targetIndex: Int)
	{
		self.values = values
		self.expected = TrainingSample.encodeOneHot(count: outputCount, target: targetIndex)
	}

}

// Extensions for sample normalization
public extension Sample
{
	
	/// Normalizes a set of samples to a range from zero to one
	///
	/// - Parameter samples: Samples which should be normalized
	/// - Returns: Normalized samples, normalization scale and offset at which the samples were normalized
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
	
	
	/// Normalizes a set of samples with a predetermined offset and factor.
	///
	/// - Parameters:
	///   - samples: Samples which should be normalized
	///   - scale: Normalization scale
	///   - offset: Normalization offset
	/// - Returns: Normalized samples
	public static func normalize(samples: [InputSample], scale: Float, offset: Float) -> [InputSample]
	{
		return samples
			.map { $0.values.values }
			.map { ($0 &+ offset) &* scale }
			// Force unwrapping because if no first element exists, this code would not be executed anyway:
			.map { Matrix3(values: $0, width: samples.first!.values.width, height: samples.first!.values.height, depth: samples.first!.values.depth) }
			.map { InputSample(values: $0) }
	}
	
	
	/// Denormalizes normalized samples back to their original ranges
	///
	/// - Parameters:
	///   - samples: Samples to denormalize
	///   - scale: Scale with which the samples were normalized
	///   - offset: Offset with which the samples were normalized
	/// - Returns: Denormalized samples.
	public static func denormalize(samples: [InputSample], scale: Float, offset: Float) -> [InputSample]
	{
		return Self.normalize(samples: samples, scale: 1 / scale, offset: -offset)
	}
	
	
	/// Creates a one-hot-matrix of width 1, height 1 and depth equal to `count`
	///
	/// - Parameters:
	///   - count: Depth of the one-hot-matrix
	///   - target: Target index which should be set to one
	/// - Returns: One-hot-matrix with a value of one at the given target index
	public static func encodeOneHot(count: Int, target: Int) -> Matrix3
	{
		precondition(count > target, "Target index greater than output length")
		var oneHotVector = [Float](repeating: 0.0, count: count)
		oneHotVector[target] = 1.0
		return Matrix3(values: oneHotVector, width: 1, height: 1, depth: count)
	}
	
}
