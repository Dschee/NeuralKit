//
//  Layers.swift
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


/// Creates a three dimensional weight matrix 
/// and fills it with small random values in the specified range
///
/// - Parameters:
///   - width: Width of the weight matrix
///   - height: Height of the weight matrix
///   - depth: Depth of the weight matrix
///   - range: Range in which the random values should be. Default: [-0.01; 0.01]
/// - Returns: Weight matrix containing random values
public func RandomWeightMatrix(width: Int, height: Int, depth: Int, range: ClosedRange<Float> = Float(-0.01) ... Float(0.01)) -> Matrix3
{
	var weightMatrix = Matrix3(repeating: 0, width: width, height: height, depth: depth)
	
	for (x,y,z) in weightMatrix.indices
	{
		weightMatrix[x,y,z] = Float(drand48()) * (range.upperBound - range.lowerBound) + range.lowerBound
	}
	
	return weightMatrix
}


/// Creates a weight matrix
/// and fills it with small random values in the specified range
///
/// - Parameters:
///   - width: Width of the weight matrix
///   - height: Height of the weight matrix
///   - range: Range in which the random values should be. Default: [-0.01; 0.01]
/// - Returns: Weight matrix containing random values
public func RandomWeightMatrix(width: Int, height: Int, range: ClosedRange<Float> = Float(-0.01) ... Float(0.01)) -> Matrix
{
	var weightMatrix = Matrix(repeating: 0, width: width, height: height)
	
	for (x,y) in weightMatrix.indices
	{
		weightMatrix[x,y] = Float(drand48()) * (range.upperBound - range.lowerBound) + range.lowerBound
	}
	
	return weightMatrix
}


/// Creates a pertubation matrix
/// consisting of very small values around zero with a few randomly larger values
///
/// - Parameters:
///   - width: Width of the perturbation matrix
///   - height: Height of the perturbation matrix
/// - Returns: Perturbation matrix
public func RandomPertubationMatrix(width: Int, height: Int) -> Matrix
{
	let matA = RandomWeightMatrix(width: width, height: height, range: 0 ... 1)
	let matB = RandomWeightMatrix(width: width, height: height, range: 0 ... 1)
	
	let transformed = matA.mapv{sqrt(-2 &* log($0))}
	let randomNegated = zip(transformed.values, matB.values).map{$1 < 0.5 ? -$0 : $0}
	return Matrix(values: randomNegated, width: width, height: height).mapv{pow($0, Array<Float>(repeating: 5, count: randomNegated.count)) &* 0.0003}
}


/// A layer of a feed forward neural network
public protocol NeuralLayer
{
	
	/// Input size of the layer.
	/// Should not change after initialization
	var inputSize: (width: Int, height: Int, depth: Int) { get }
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	var outputSize: (width: Int, height: Int, depth: Int) { get }
	
	
	/// Calculates the activation function for all inputs of the layer
	///
	/// - Parameter input: Layer input
	/// - Returns: Result of the activation function
	func activated(_ input: Matrix3) -> Matrix3
	
	
	/// Weighs the outputs of the activation function so it can be presented
	/// to the next layer
	///
	/// - Parameter output: Output of the activation function which should be forwarded
	/// - Returns: Weighted output of the layer
	func weighted(_ output: Matrix3) -> Matrix3
	
	
	/// Adjusts the weights of the layer to reduce the error of the network.
	///
	/// The errors of the next layer will be provided.
	/// The function has to also calculate the errors of the layer.
	///
	///
	/// - Parameters:
	///   - nextLayerErrors: Error matrix from the input of the next layer
	///   - outputs: Outputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	/// - Returns: Error matrix of the current layer
	mutating func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float) -> Matrix3
	
}

// Feed forward function
public extension NeuralLayer
{
	
	/// Forwards an input through the layer.
	/// 
	/// The inputs will be fed through the activation function
	/// and will be weighted.
	///
	/// - Parameter input: Layer input
	/// - Returns: Next layer input
	func forward(_ input: Matrix3) -> Matrix3
	{
		return weighted(activated(input))
	}
}


/// A fully connected layer of a neural network.
/// All neurons of the layer are connected to all neurons of the next layer
public struct FullyConnectedLayer: NeuralLayer
{
	
	/// Input size of the layer.
	/// Should not change after initialization
	public var inputSize: (width: Int, height: Int, depth: Int)
	{
		// The last neuron is an extra bias neuron and will not be counted in input depth
		return (width: 1, height: 1, depth: weights.width - 1)
	}
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return (width: 1, height: 1, depth: weights.height)
	}
	
	
	/// Weights with which outputs of the layer are weighted when presented to the next layer
	public private(set) var weights: Matrix
	
	
	/// Activation function which will be applied to the inputs of the layer
	public let activationFunction: ([Float]) -> [Float]
	
	
	/// Derivative of the activation function used for training the network.
	public let activationDerivative: ([Float]) -> [Float]
	
	
	/// Initializes a fully connected neural layer using the given weight matrix, its activation function and derivative
	///
	/// The value at column n and row m of the weight matrix corresponds to the weight of the nth neuron 
	/// towards the mth neuron of the next layer.
	///
	/// **Note:** The input size of the layer will be one neuron smaller than the width of the weight matrix
	/// as the weight matrix will also store bias values of the layer
	///
	/// - Parameters:
	///   - weights: Weights from the layer to the next layer
	///   - activationFunction: Activation function with which the inputs should be activated
	///   - activationDerivative: Derivative of the activation function used for training
	public init(weights: Matrix, activationFunction: @escaping ([Float]) -> [Float], activationDerivative: @escaping ([Float]) -> [Float])
	{
		self.weights = weights
		self.activationFunction = activationFunction
		self.activationDerivative = activationDerivative
	}
	
	
	/// Calculates the activation function for all inputs of the layer
	///
	/// - Parameter input: Layer input
	/// - Returns: Result of the activation function
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return Matrix3(values: (activationFunction(input.values) + [1]), width: inputSize.width, height: inputSize.height, depth: inputSize.depth+1)
	}
	
	
	/// Weighs the outputs of the activation function so it can be presented
	/// to the next layer
	///
	/// - Parameter output: Output of the activation function which should be forwarded
	/// - Returns: Weighted output of the layer
	public func weighted(_ output: Matrix3) -> Matrix3
	{
		return Matrix3(values: weights * output.values, width: 1, height: 1, depth: weights.height)
	}
	
	
	/// Adjusts the weights of the layer to reduce the error of the network.
	///
	/// The errors of the next layer will be provided.
	/// The function has to also calculate the errors of the layer.
	///
	///
	/// - Parameters:
	///   - nextLayerErrors: Error matrix from the input of the next layer
	///   - outputs: Outputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	/// - Returns: Error matrix of the current layer
	public mutating func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float) -> Matrix3
	{
		// Calculating signal errors
		let weightedErrors = weights.transposed * nextLayerErrors.values
		let errorsIncludingBias = weightedErrors &* (activationDerivative(outputs.values))
		
		// Transforming data for outer vector product
		let nextLayerErrorVector = Matrix(values: nextLayerErrors.values, width: 1, height: nextLayerErrors.depth)
		let outVector = Matrix(values: outputs.values, width: outputs.depth, height: 1)
		
		// Adjusting weights by calculating weight delta matrix
		let weightDelta = (nextLayerErrorVector * outVector).mapv{$0 &* learningRate}
		
		// Applying weight change.
		weights = weights + weightDelta
		
		// Simulated annealing (helps overcome local minima)
		if annealingRate != 0
		{
			weights = weights + RandomPertubationMatrix(width: weights.width, height: weights.height).mapv{$0 &* annealingRate}
		}
		
		// Bias error is dropped.
		return Matrix3(values: Array<Float>(errorsIncludingBias.dropLast()), width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
	}

}

public struct ConvolutionLayer: NeuralLayer
{
	public private(set) var kernels: [Matrix]
	public var bias: [Float]
	
	public let inputSize: (width: Int, height: Int, depth: Int)
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return (
			width: inputSize.width - (kernels.first?.width ?? 0),
			height: inputSize.height - (kernels.first?.height ?? 0),
			depth: kernels.count
		)
	}

	// TODO: insets and strides
//	public let horizontalStride: Int
//	public let verticalStride: Int
//	
//	public let horizontalInset: Int
//	public let verticalInset: Int
	
	public let activationFunction: ([Float]) -> [Float]
	public let activationDerivative: ([Float]) -> [Float]
	

	public func activated(_ input: Matrix3) -> Matrix3
	{
		return input.mapv(activationFunction)
	}
	
	public func weighted(_ activated: Matrix3) -> Matrix3
	{
		var output = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		
		for (x,y,z) in output.indices
		{
			let kernel = kernels[z]
			output[x,y,z] = (0 ..< activated.depth)
				.map{activated[x:x, y:y, z:$0, width:kernel.width, height:kernel.height, depth: 1]}
				.map{$0.values}
				.map{$0 * kernel.values}
				.reduce(0,+)
		}
		
		return output
	}
	
	public mutating func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float) -> Matrix3
	{
		var errors = Matrix3(repeating: 0, width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
		
		for z in 0 ..< inputSize.depth
		{
			for y in 0 ..< inputSize.height
			{
				for x in 0 ..< inputSize.width
				{
					
				}
			}
		}
		
		fatalError()
	}
}

public struct PoolingLayer: NeuralLayer
{
	public let inputSize: (width: Int, height: Int, depth: Int)
	public let outputSize: (width: Int, height: Int, depth: Int)
	
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
	
	public func adjustWeights(nextLayerErrors: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float) -> Matrix3
	{
		
		fatalError()
	}
}
