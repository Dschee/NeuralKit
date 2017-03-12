//
//  Samples.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 19.02.17.
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
	
	
	/// Initializes an input sample with the given value matrix.
	/// The dimensions of the value matrix must fit the input dimensions
	/// of a neural network which will process the sample
	///
	/// - Parameter values: Input values
	public init(values: Matrix3)
	{
		self.values = values
	}
	
}


/// A training sample which also provides an expected output
public struct TrainingSample: Sample
{
	
	/// Values of the sample in a three dimensional matrix
	public let values: Matrix3
	
	
	/// Expected output values towards which a network can be trained
	public let expected: Matrix3
	
	
	/// Initializes a training sample which can be used to train a neural
	/// network using backpropagation.
	///
	/// The training sample stores an input aswell as the output which is
	/// expected for the given input.
	///
	/// The input value must match the network input dimensions.
	/// The output value must match the network output dimensions.
	///
	/// - Parameters:
	///   - values: Input value for the network.
	///   - expected: Expected output for the given input
	public init (values: Matrix3, expected: Matrix3)
	{
		self.values = values
		self.expected = expected
	}
	
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
	public init(values: Matrix3, outputCount: Int, targetIndex: Int, baseValue: Float = 0.0, hotValue: Float = 1.0)
	{
		self.values = values
		self.expected = TrainingSample.encodeOneHot(count: outputCount, target: targetIndex, baseValue: baseValue, hotValue: hotValue)
	}

}

// Extensions for sample normalization
public extension Sample
{
	
	/// Normalizes a set of samples to a range from zero to one
	///
	/// - Parameter samples: Samples which should be normalized
	/// - Returns: Normalized samples, normalization scale and offset at which the samples were normalized
	public static func normalize(samples: [Matrix3]) -> ([Matrix3], scale: Float, offset: Float)
	{
		guard
			let min = samples.flatMap({$0.values.min()}).min(),
			let max = samples.flatMap({$0.values.max()}).max()
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
	public static func normalize(samples: [Matrix3], scale: Float, offset: Float) -> [Matrix3]
	{
		return samples
			.map { $0.values }
			.map { ($0 &+ offset) &* scale }
			// Force unwrapping because if no first element exists, this code would not be executed anyway:
			.map { Matrix3(values: $0, width: samples.first!.width, height: samples.first!.height, depth: samples.first!.depth) }
	}
	
	
	/// Denormalizes normalized samples back to their original ranges
	///
	/// - Parameters:
	///   - samples: Samples to denormalize
	///   - scale: Scale with which the samples were normalized
	///   - offset: Offset with which the samples were normalized
	/// - Returns: Denormalized samples.
	public static func denormalize(samples: [Matrix3], scale: Float, offset: Float) -> [Matrix3]
	{
		return Self.normalize(samples: samples, scale: 1 / scale, offset: -offset)
	}
	
	
	/// Creates a one-hot-matrix of width 1, height 1 and depth equal to `count`
	///
	/// - Parameters:
	///   - count: Depth of the one-hot-matrix
	///   - target: Target index which should be set to one
	/// - Returns: One-hot-matrix with a value of one at the given target index
	public static func encodeOneHot(count: Int, target: Int, baseValue: Float = 0.0, hotValue: Float = 1.0) -> Matrix3
	{
		precondition(count > target, "Target index greater than output length")
		
		var oneHotVector = [Float](repeating: baseValue, count: count)
		oneHotVector[target] = hotValue
		return Matrix3(values: oneHotVector, width: 1, height: 1, depth: count)
	}
	
	
	/// Splits a set of training samples randomly into a training and a testing set.
	///
	/// The ratio determines how many of the samples will be in the training set.
	///
	/// - Parameters:
	///   - samples: Training set to split
	///   - ratio: Ratio of samples which will be in the training set
	/// - Returns: Two sample sets. The first contains `ratio*samples.count` elements,
	/// the second contains `(1-ratio)*samples.count` elements.
	public static func split(samples: [TrainingSample], ratio: Float) -> ([TrainingSample], [TrainingSample])
	{
		let trainingSetCount = Int(Float(samples.count) * ratio)
		
		var trainingSamples:[TrainingSample] = []
		var testingSamples = samples
		
		for _ in 0 ..< trainingSetCount
		{
			let index = Int(arc4random_uniform(UInt32(testingSamples.count)))
			trainingSamples.append(testingSamples.remove(at: index))
		}
		
		return (trainingSamples, testingSamples)
	}
	
}

