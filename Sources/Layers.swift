//
//  Layers.swift
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

public func RandomWeightMatrix(width: Int, height: Int, depth: Int, range: ClosedRange<Float> = Float(-0.01) ... Float(0.01)) -> Matrix3
{
	var weightMatrix = Matrix3(repeating: 0, width: width, height: height, depth: depth)
	
	for (x,y,z) in weightMatrix.indices
	{
		weightMatrix[x,y,z] = Float(drand48()) * (range.upperBound - range.lowerBound) + range.lowerBound
	}
	
	return weightMatrix
}

public func RandomWeightMatrix(width: Int, height: Int, range: ClosedRange<Float> = Float(-0.01) ... Float(0.01)) -> Matrix
{
	var weightMatrix = Matrix(repeating: 0, width: width, height: height)
	
	for (x,y) in weightMatrix.indices
	{
		weightMatrix[x,y] = Float(drand48()) * (range.upperBound - range.lowerBound) + range.lowerBound
	}
	
	return weightMatrix
}

public protocol NeuralLayer
{
	var inputSize: (width: Int, height: Int, depth: Int) { get }
	var outputSize: (width: Int, height: Int, depth: Int) { get }
	
	func activated(_ input: Matrix3) -> Matrix3
	func weighted(_ output: Matrix3) -> Matrix3
	
	mutating func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float) -> Matrix3
}

public extension NeuralLayer
{
	func forward(_ input: Matrix3) -> Matrix3
	{
		return weighted(activated(input))
	}
}

public struct FullyConnectedLayer: NeuralLayer
{
	public var inputSize: (width: Int, height: Int, depth: Int)
	{
		// The last neuron is an extra bias neuron and will not be counted in input depth
		return (width: 1, height: 1, depth: weights.width - 1)
	}
	
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return (width: 1, height: 1, depth: weights.height)
	}
	
	public private(set) var weights: Matrix
	public let activationFunction: ([Float]) -> [Float]
	public let activationDerivative: ([Float]) -> [Float]
	
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return Matrix3(values: (activationFunction(input.values) + [1]), width: inputSize.width, height: inputSize.height, depth: inputSize.depth+1)
	}
	
	public func weighted(_ output: Matrix3) -> Matrix3
	{
		return Matrix3(values: weights * output.values, width: 1, height: 1, depth: weights.height)
	}
	
//	public func forward(_ input: Matrix3) -> Matrix3
//	{
//		precondition(input.dimension == inputSize, "Size of input sample must match layer input size")
//		// + [1] appends bias value to input vector
//		return Matrix3(values: weights * (self.activationFunction(input.values) + [1]), width: 1, height: 1, depth: weights.height)
//	}
	
	public mutating func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float) -> Matrix3
	{
		// Calculating signal errors
		let weightedErrors = weights.transposed * nextLayerErrors.values
		let errorsIncludingBias = weightedErrors &* (activationDerivative(outputs.values))
		
		
		
		// Bias error is dropped.
		return Matrix3(values: Array<Float>(errorsIncludingBias.dropLast()), width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
	}
	
//	func calculateSignalErrors(expectedOutputs: [Float], actualOutputs: [Float]) -> [Float]
//	{
//		return (expectedOutputs &- actualOutputs) &* activationDerivative(actualOutputs)
//	}
//	
//	public func layerErrors(forNextLayerErrors nextLayerErrors: Matrix3, layerOutput: Matrix3) -> Matrix3
//	{
//		let weightedErrors = weights.transposed * nextLayerErrors.values
//		let errors = weightedErrors &* activationDerivative(layerOutput.values)
//		
//		return Matrix3(values: errors, width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
//	}
}

public struct ConvolutionLayer: NeuralLayer
{
	public private(set) var kernels: [Matrix3]
	public var bias: [Float]
	
	public let inputSize: (width: Int, height: Int, depth: Int)
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return (
			width: (inputSize.width - 2 * horizontalInset) / horizontalStride,
			height: (inputSize.height - 2 * verticalInset) / verticalStride,
			depth: kernels.count
		)
	}
	
	public let horizontalStride: Int
	public let verticalStride: Int
	
	public let horizontalInset: Int
	public let verticalInset: Int
	
	public let activationFunction: ([Float]) -> [Float]
	public let activationDerivative: ([Float]) -> [Float]
	
//	public func forward(_ input: Matrix3) -> Matrix3
//	{
//		precondition(!kernels.isEmpty, "Convolution layer must have at least one convolution kernel")
//		
//		var output = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
//		
//		for y in 0 ..< outputSize.height
//		{
//			let inputY = y * verticalStride + verticalInset
//			for x in 0 ..< outputSize.width
//			{
//				let inputX = x * horizontalStride + horizontalInset
//				let slice = input[x: inputX, y: inputY, z: 0, width: kernels.first!.width, height: kernels.first!.height, depth: input.depth]
//				
//				for (z, kernel) in kernels.enumerated()
//				{
//					output[x, y, z] = kernel.convolve(with: slice)
//				}
//			}
//		}
//		
//		output.values = activationFunction(output.values)
//		return output
//	}
	
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return Matrix3(values: activationFunction(input.values), width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
	}
	
	public func weighted(_ output: Matrix3) -> Matrix3
	{
		fatalError()
	}
	
	public mutating func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float) -> Matrix3
	{
//		var errors = Matrix3(repeating: 0, width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
//		
//		for y in 0 ..< outputSize.height
//		{
//			for x in 0 ..< outputSize.width
//			{
//				
//			}
//		}
//		
//		return errors
		fatalError("TODO")
	}
}

public struct PoolingLayer: NeuralLayer
{
	public let inputSize: (width: Int, height: Int, depth: Int)
	public let outputSize: (width: Int, height: Int, depth: Int)
	
//	public func forward(_ input: Matrix3) -> Matrix3
//	{
//		precondition(inputSize.width  % outputSize.width  == 0, "Scaling factor from output to input must be an integer")
//		precondition(inputSize.height % outputSize.height == 0, "Scaling factor from output to input must be an integer")
//		precondition(inputSize.depth  % outputSize.depth  == 0, "Scaling factor from output to input must be an integer")
//		
//	}
	
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return input
	}
	
	public func weighted(_ activated: Matrix3) -> Matrix3
	{
		let xScale = inputSize.width / outputSize.width
		let yScale = inputSize.height / outputSize.height
		let zScale = inputSize.depth / outputSize.depth
		
		var output = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		
		for z in 0 ..< outputSize.depth
		{
			let zOffset = z * zScale
			for y in 0 ..< outputSize.height
			{
				let yOffset = y * yScale
				for x in 0 ..< outputSize.width
				{
					let xOffset = x * xScale
					
					let submatrix = activated[x: xOffset, y: yOffset, z: zOffset, width: xScale, height: yScale, depth: zScale]
					let max = submatrix.values.max() ?? 0
					output[x, y, z] = max
				}
			}
		}
		return output
	}
	
	public func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float) -> Matrix3
	{
		fatalError()
	}
}
