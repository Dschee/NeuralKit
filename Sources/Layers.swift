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
import Accelerate


/// Activation functions for a neural network
///
/// - sigmoid: logistic growth from 0 to 1
/// - tanh: logistic growth from -1 to 1
/// - relu: 0 for negative inputs, identity for positive inputs
/// - linear: identity
public enum Activation
{
	/// Logistic growth from 0 to 1
	case sigmoid
	
	/// Logistic growth from -1 to 1
	case tanh
	
	/// max(0, input)
	case relu
	
	/// identity (output = input)
	case linear
	
	/// softmax activation function
	case softmax
}


// Extension for retrieving the actual function or its derivative
internal extension Activation
{
	
	/// Returns a vectorized activation function
	var function: ([Float]) -> [Float]
	{
		switch self
		{
		case .sigmoid:
			return NeuralKit.sigmoid
			
		case .tanh:
			return NeuralKit.tanh
			
		case .relu:
			return NeuralKit.relu
			
		case .linear:
			return NeuralKit.identity
			
		case .softmax:
			return NeuralKit.softmax
		}
	}
	
	/// Returns a vectorized derivative of the activation function
	var derivative: ([Float]) -> [Float]
	{
		switch self
		{
		case .sigmoid:
			return NeuralKit.sigmoid_deriv
			
		case .tanh:
			return NeuralKit.tanh_deriv
			
		case .relu:
			return NeuralKit.relu_deriv
			
		case .linear:
			return NeuralKit.ones
			
		case .softmax:
			return NeuralKit.softmax_deriv
		}
	}
}


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
	
	
	/// Adjusts the weights of the layer to reduce the total error of the network 
	/// using gradient descent.
	///
	/// The gradients of the posterior layer will be provided.
	/// The function has to also calculate the errors of the layer 
	/// for the anterior layer to use.
	///
	/// - Parameters:
	///   - nextLayerGradients: Error matrix from the input of the next layer
	///   - outputs: Outputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	mutating func adjustWeights(
		nextLayerGradients: Matrix3,
		inputs: Matrix3,
		outputs: Matrix3,
		learningRate: Float,
		annealingRate: Float,
		momentum: Float,
		decay: Float
	) -> Matrix3
	
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
	public internal(set) var weights: Matrix
	
	/// Weight delta for momentum training
	private var weightDelta: Matrix
	
	
	/// Activation function which will be applied to the inputs of the layer
	public let activationFunction: Activation
	
	
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
	public init(weights: Matrix, activationFunction: Activation)
	{
		self.weights = weights
		self.weightDelta = Matrix(repeating: 0, width: weights.width, height: weights.height)
		self.activationFunction = activationFunction
	}
	
	
	/// Calculates the activation function for all inputs of the layer
	///
	/// - Parameter input: Layer input
	/// - Returns: Result of the activation function
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return Matrix3(values: (activationFunction.function(input.values) + [1]), width: inputSize.width, height: inputSize.height, depth: inputSize.depth+1)
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
	
	
	/// Adjusts the weights of the layer to reduce the total error of the network
	/// using gradient descent.
	///
	/// The gradients of the posterior layer will be provided.
	/// The function has to also calculate the errors of the layer
	/// for the anterior layer to use.
	///
	/// - Parameters:
	///   - nextLayerGradients: Error matrix from the input of the next layer
	///   - outputs: Outputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public mutating func adjustWeights(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float, momentum: Float, decay: Float) -> Matrix3
	{
		// Calculating signal errors
		let weightedErrors = Matrix.multiply(weights, nextLayerGradients.values, transpose: true)
		let errorsIncludingBias = weightedErrors &* (activationFunction.derivative(outputs.values))
		
		// Transforming data for outer vector product
//		let nextLayerErrorVector = Matrix(values: nextLayerGradients.values, width: 1, height: nextLayerGradients.depth)
//		let outVector = Matrix(values: outputs.values, width: outputs.depth, height: 1)
//		
//		// Adjusting weights by calculating weight delta matrix
//		let weightDelta = (nextLayerErrorVector * outVector).mapv{$0 &* learningRate}
//		
//		// Applying weight change.
//		weights = weights + weightDelta
		
		weights.addMultiplied(
			Matrix(
				nextLayerGradients.reshaped(
					width: 1,
					height: nextLayerGradients.depth,
					depth: 1
				)
			),
			Matrix(
				outputs.reshaped(
					width: outputs.depth,
					height: 1,
					depth: 1
				)
			),
			factor: learningRate
		)
		
		// Applying momentum (helps overcome local minima)
		if momentum != 0
		{
			self.weights += self.weightDelta
			self.weightDelta.add(weightDelta, factor: momentum)
		}
		
		// Simulated annealing (helps overcome local minima)
		if annealingRate != 0
		{
			weights.add(
				RandomPertubationMatrix(width: weights.width, height: weights.height),
				factor: annealingRate
			)
		}
		
		// Applying weight decay to keep weights small
		if decay != 0
		{
			weights *= (1 - decay)
		}
		
		// Bias error is dropped.
		return Matrix3(values: Array<Float>(errorsIncludingBias.dropLast()), width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
	}

}


/// A layer which is connected to the next layer using convolution kernels.
public struct ConvolutionLayer: NeuralLayer
{
	
	/// The convolution kernels which are applied to input values
	public private(set) var kernels: [Matrix3]
	
	
	/// The bias values which are applied after convolutions
	public var bias: [Float]
	
	
	/// Input size of the layer.
	/// Should not change after initialization
	public let inputSize:  (width: Int, height: Int, depth: Int)
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		let kernelWidth = kernels.first?.width ?? 0
		let kernelHeight = kernels.first?.height ?? 0
		
		let stridedWidth = inputSize.width / horizontalStride
		let stridedHeight = inputSize.height / verticalStride
		
		return (
			width:  stridedWidth  - kernelWidth + 1 - (2 * horizontalInset),
			height: stridedHeight - kernelHeight + 1 - (2 * verticalInset),
			depth:  kernels.count
		)
	}

	
	/// Stride at which the input matrix is traversed horizontally
	public let horizontalStride: Int
	
	
	/// Stride at which the input matrix is traversed vertically
	public let verticalStride: Int
	
	
	/// Horizontal inset at which the traversion of the input matrix begins and ends
	public let horizontalInset: Int
	
	
	/// Vertical inset at which the traversion of the input matrix begins and ends
	public let verticalInset: Int
	
	
	/// Activation function applied on each neuron
	public let activationFunction: Activation
	
	
	/// Calculates the activation function for all inputs of the layer
	///
	/// - Parameter input: Layer input
	/// - Returns: Result of the activation function
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return input.mapv(activationFunction.function)
	}
	
	
	/// Weighs the outputs of the activation function so it can be presented
	/// to the next layer
	///
	/// - Parameter output: Output of the activation function which should be forwarded
	/// - Returns: Weighted output of the layer
	public func weighted(_ activated: Matrix3) -> Matrix3
	{
		var output = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)

		for (z, kernel) in kernels.enumerated()
		{
			let convolved = activated.convolved(
				with: kernel,
				horizontalStride: horizontalStride,
				verticalStride: verticalStride,
				lateralStride: 1,
				horizontalInset: horizontalInset,
				verticalInset: verticalInset,
				lateralInset: 0
			)
			output[x: 0, y: 0, z: z, width: output.width, height: output.height, depth: output.depth] = convolved.mapv{$0 &+ bias[z]}
		}
		
		return output
	}
	
	
	/// Adjusts the weights of the layer to reduce the total error of the network
	/// using gradient descent.
	///
	/// The gradients of the posterior layer will be provided.
	/// The function has to also calculate the errors of the layer
	/// for the anterior layer to use.
	///
	/// - Parameters:
	///   - nextLayerGradients: Error matrix from the input of the next layer
	///   - outputs: Outputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public mutating func adjustWeights(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float, momentum: Float, decay: Float) -> Matrix3
	{
		var gradient = Matrix3(repeating: 0, width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)

//		for (z, kernel) in kernels.enumerated()
//		{
//			let source = nextLayerGradients[
//				x: 0,
//				y: 0,
//				z: z,
//				width: outputSize.width,
//				height: outputSize.height,
//				depth: 1
//			]
//			let correlated = source.correlated(
//				with: kernel,
//				horizontalStride: horizontalStride,
//				verticalStride: verticalStride,
//				lateralStride: 1,
//				horizontalInset: horizontalInset,
//				verticalInset: verticalInset,
//				lateralInset: 0
//			)
//			gradient += correlated
//		}
		
		for (outputZ, kernel) in kernels.enumerated()
		{
			var kernelGradient = Matrix3.init(repeating: 0, width: kernel.width, height: kernel.height, depth: kernel.depth)
			
			for outputY in 0 ..< outputSize.height
			{
				for outputX in 0 ..< outputSize.width
				{
					let outputGradient = nextLayerGradients[outputX, outputY, outputZ]
					
					for (x,y,z) in kernel.indices
					{
						
					}
				}
			}
			
			kernels[outputZ] += kernelGradient.mapv{$0 &* learningRate}
		}
		
		return gradient
	}
	
}

/// A pooling layer for reducing dimensionality using max pooling.
public struct PoolingLayer: NeuralLayer
{
	
	/// Input size of the layer.
	/// Should not change after initialization
	public let inputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	public let outputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Creates a new Max-Pooling layer with the given input and output size.
	/// Scaling factors are determined automatically from the input and output size.
	///
	/// - Parameters:
	///   - inputSize: Size of the layer input
	///   - outputSize: Size of the layer output
	public init(inputSize: (width: Int, height: Int, depth: Int), outputSize: (width: Int, height: Int, depth: Int))
	{
		self.inputSize = inputSize
		self.outputSize = outputSize
	}
	
	
	/// Calculates the activation function for all inputs of the layer
	///
	/// - Parameter input: Layer input
	/// - Returns: Result of the activation function
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return input
	}
	
	
	/// Weighs the outputs of the activation function so it can be presented
	/// to the next layer
	///
	/// - Parameter output: Output of the activation function which should be forwarded
	/// - Returns: Weighted output of the layer
	public func weighted(_ activated: Matrix3) -> Matrix3
	{
		let xScale = inputSize.width / outputSize.width
		let yScale = inputSize.height / outputSize.height
		let zScale = inputSize.depth / outputSize.depth
		
		var output = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		
		for (x,y,z) in output.indices
		{
			let xOffset = x * xScale
			let yOffset = y * yScale
			let zOffset = z * zScale
			
			let submatrix = activated[x: xOffset, y: yOffset, z: zOffset, width: xScale, height: yScale, depth: zScale]
			let max = submatrix.values.max() ?? 0
			output[x, y, z] = max
		}
		
		return output
	}
	
	
	/// Adjusts the weights of the layer to reduce the total error of the network
	/// using gradient descent.
	///
	/// The gradients of the posterior layer will be provided.
	/// The function has to also calculate the errors of the layer
	/// for the anterior layer to use.
	///
	/// - Parameters:
	///   - nextLayerGradients: Error matrix from the input of the next layer
	///   - outputs: Outputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public func adjustWeights(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float, momentum: Float, decay: Float) -> Matrix3
	{
		let xScale = inputSize.width / outputSize.width
		let yScale = inputSize.height / outputSize.height
		let zScale = inputSize.depth / outputSize.depth
		
		var gradients = Matrix3(repeating: 0, width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
		
		for (x, y, z) in gradients.indices where inputs[x, y, z] == outputs[x / xScale, y / yScale, z / zScale]
		{
			// Only the maximum inputs get the gradient from the next layer, everything else is zero.
			gradients[x, y, z] = nextLayerGradients[x / xScale, y / yScale, z / zScale]
		}
		
		return gradients
	}
	
}


/// A layer which reshapes the output of one layer to fit the input of another layer
public struct ReshapingLayer: NeuralLayer
{
	
	/// Output size of the layer.
	/// Should not change after initialization
	public var outputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Input size of the layer.
	/// Should not change after initialization
	public var inputSize: (width: Int, height: Int, depth: Int)

	
	/// Initializes a reshaping layer which
	/// reshapes the output of one layer to fit the input of another layer.
	///
	/// The number of values stored in the input to this layer must match
	/// the number of outputs of this layer
	///
	/// - Parameters:
	///   - inputSize: Size of the input matrix
	///   - outputSize: Size of the output matrix
	public init(inputSize: (width: Int, height: Int, depth: Int), outputSize: (width: Int, height: Int, depth: Int))
	{
		precondition(
			inputSize.width * inputSize.height * inputSize.depth ==
			outputSize.width * outputSize.height * outputSize.depth,
			"Input and outputs size must have matching size."
		)
		self.inputSize = inputSize
		self.outputSize = outputSize
	}
	
	
	/// Calculates the activation function for all inputs of the layer
	///
	/// - Parameter input: Layer input
	/// - Returns: Result of the activation function
	public func activated(_ input: Matrix3) -> Matrix3
	{
		return input
	}
	
	
	/// Weighs the outputs of the activation function so it can be presented
	/// to the next layer
	///
	/// - Parameter output: Output of the activation function which should be forwarded
	/// - Returns: Weighted output of the layer
	public func weighted(_ output: Matrix3) -> Matrix3
	{
		return output.reshaped(width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
	}
	
	
	/// Adjusts the weights of the layer to reduce the total error of the network
	/// using gradient descent.
	///
	/// The gradients of the posterior layer will be provided.
	/// The function has to also calculate the errors of the layer
	/// for the anterior layer to use.
	///
	/// - Parameters:
	///   - nextLayerGradients: Error matrix from the input of the next layer
	///   - outputs: Outputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public mutating func adjustWeights(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3, learningRate: Float, annealingRate: Float, momentum: Float, decay: Float) -> Matrix3
	{
		return nextLayerGradients.reshaped(width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
	}
	
}
