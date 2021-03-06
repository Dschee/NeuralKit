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
import MatrixVector


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
			return MatrixVector.sigmoid
			
		case .tanh:
			return MatrixVector.tanh
			
		case .relu:
			return MatrixVector.relu
			
		case .linear:
			return MatrixVector.identity
			
		case .softmax:
			return MatrixVector.softmax
		}
	}
	
	/// Returns a vectorized derivative of the activation function
	var derivative: ([Float]) -> [Float]
	{
		switch self
		{
		case .sigmoid:
			return MatrixVector.sigmoid_deriv
			
		case .tanh:
			return MatrixVector.tanh_deriv
			
		case .relu:
			return MatrixVector.relu_deriv
			
		case .linear:
			return MatrixVector.ones
			
		case .softmax:
			return MatrixVector.softmax_deriv
		}
	}
}


/// A tensor is a a*...*n dimensional vector/matrix/etc.
///
/// - vector: A one dimensional vector
/// - matrix: A two dimensional matrix
/// - matrix3: A three dimensional tensor
public enum Tensor
{
	case vector([Float])
	
	case matrix(Matrix)
	
	case matrix3(Matrix3)
}


public extension Tensor
{
	
	/// Retrieves the values which a tensor stores or sets them.
	public var values: [Float]
	{
		get
		{
			switch self
			{
			case .vector(let values):
				return values
				
			case .matrix(let matrix):
				return matrix.values
				
			case .matrix3(let matrix):
				return matrix.values
			}
		}
		
		set (new)
		{
			switch self
			{
			case .vector(_):
				self = .vector(new)
				
			case .matrix(let matrix):
				self = .matrix(Matrix(values: new, width: matrix.width, height: matrix.height))
				
			case .matrix3(let matrix):
				self = .matrix3(Matrix3(values: new, width: matrix.width, height: matrix.height, depth: matrix.depth))
			}
		}
	}
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
	
	
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	func forward(_ input: Matrix3) -> Matrix3
	
}


public protocol BidirectionalLayer: NeuralLayer
{
	/// Adjusts the weights of the layer to reduce the total error of the network
	/// using gradient descent.
	///
	/// The gradients of the posterior layer will be provided.
	/// The function has to also calculate the errors of the layer
	/// for the anterior layer to use.
	///
	/// - Parameters:
	///   - nextLayerGradients: Error matrix from the input of the next layer
	///   - inputs: Inputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	mutating func updateGradients(
		nextLayerGradients: Matrix3,
		inputs: Matrix3,
		outputs: Matrix3
	) -> Matrix3
}


public protocol OutputLayer: NeuralLayer
{
	func loss(
		expected: Matrix3,
		output: Matrix3
	) -> Matrix3
}

public protocol WeightAdjustableLayer: BidirectionalLayer
{
	var weights: [Tensor] { get set }
	var weightGradients: [Tensor] { get set }
}


/// A fully connected layer of a neural network.
/// All neurons of the layer are connected to all neurons of the next layer
public struct FullyConnectedLayer: WeightAdjustableLayer
{
	/// Input size of the layer.
	/// Should not change after initialization
	public var inputSize: (width: Int, height: Int, depth: Int)
	{
		// The last neuron is an extra bias neuron and will not be counted in input depth
		return (width: 1, height: 1, depth: weightMatrix.width - 1)
	}
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return (width: 1, height: 1, depth: weightMatrix.height)
	}
	
	
	/// Weights with which outputs of the layer are weighted when presented to the next layer
	public internal(set) var weightMatrix: Matrix
	
	var weightGradientMatrix: Matrix
	
	
	public var weightGradients: [Tensor]
	{
		get
		{
			return [.matrix(weightGradientMatrix)]
		}
		
		set (new)
		{
			guard case .matrix(let matrix) = new[0] else { fatalError() }
			weightGradientMatrix = matrix
		}
	}
	
	
	public var weights: [Tensor]
	{
		get
		{
			return [.matrix(weightMatrix)]
		}
		
		set (new)
		{
			guard case .matrix(let matrix) = new[0] else { fatalError() }
			weightMatrix = matrix
		}
	}
	
	
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
	public init(weights: Matrix)
	{
		self.weightMatrix = weights
		self.weightGradientMatrix = Matrix(repeating: 0, width: weights.width, height: weights.height)
	}
	
	
	/// Initializes a new fully connected layer with the given input and output depth
	/// and fills its weight matrix with random small values.
	///
	/// - Parameters:
	///   - inputDepth: Depth of the input volume. Width and height must be zero.
	///   - outputDepth: Depth of the output volume. Width and height will be zero.
	public init(inputDepth: Int, outputDepth: Int)
	{
		self.weightMatrix = RandomWeightMatrix(width: inputDepth + 1, height: outputDepth)
		self.weightGradientMatrix = Matrix(repeating: 0, width: inputDepth + 1, height: outputDepth)
	}
	
	
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	public func forward(_ input: Matrix3) -> Matrix3
	{
		return Matrix3(values: weightMatrix * (input.values + [1]), width: 1, height: 1, depth: weightMatrix.height)
	}
	
	
	public mutating func updateGradients(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3) -> Matrix3
	{
		// Calculating signal errors for anterior layer
		let errorsIncludingBias = Matrix.multiply(weightMatrix, nextLayerGradients.values, transpose: true)

		weightGradientMatrix.addMultiplied(
			Matrix(
				nextLayerGradients.reshaped(
					width: 1,
					height: nextLayerGradients.depth,
					depth: 1
				)
			),
			Matrix(
				values: inputs.values + [1],
				width: inputSize.depth + 1,
				height: 1
			),
			destinationFactor: 1
		)
		
		// Bias error is dropped.
		return Matrix3(
			values: Array<Float>(errorsIncludingBias.dropLast()),
			width: self.inputSize.width,
			height: self.inputSize.height,
			depth: self.inputSize.depth
		)
	}
	
}


/// A layer which is connected to the next layer using convolution kernels.
public struct ConvolutionLayer: WeightAdjustableLayer
{
	/// The convolution kernels which are applied to input values
	public private(set) var kernels: [Matrix3]
	
	
	public private(set) var kernelGradients: [Matrix3]
	
	
	/// The bias values which are applied after convolutions
	public private(set) var bias: [Float]
	
	
	public private(set) var biasGradients: [Float]
	
	
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
	
	
	public var weights: [Tensor]
	{
		get
		{
			return kernels.map{.matrix3($0)} + [.vector(bias)]
		}
		
		set (new)
		{
			for (index, tensor) in new.enumerated()
			{
				guard case .matrix3(let matrix) = tensor else { fatalError() }
				kernels[index] = matrix
			}
			
			guard case .vector(let biasVector) = new.last! else { fatalError() }
			bias = biasVector
		}
	}
	
	public var weightGradients: [Tensor]
	{
		get
		{
			return kernelGradients.map{.matrix3($0)} + [.vector(biasGradients)]
		}
		
		set (new)
		{
			for (index, tensor) in new.enumerated()
			{
				guard case .matrix3(let matrix) = tensor else { fatalError() }
				kernelGradients[index] = matrix
			}
			
			guard case .vector(let biasGradientVector) = new.last! else { fatalError() }
			biasGradients = biasGradientVector
		}
	}
	
	
	/// Stride at which the input matrix is traversed horizontally
	public let horizontalStride: Int
	
	
	/// Stride at which the input matrix is traversed vertically
	public let verticalStride: Int
	
	
	/// Horizontal inset at which the traversion of the input matrix begins and ends
	public let horizontalInset: Int
	
	
	/// Vertical inset at which the traversion of the input matrix begins and ends
	public let verticalInset: Int
	
	
	/// Initializes a new convolutional layer.
	///
	/// A convolutional layer applies a set of given convolution filters
	/// to a provided input.
	///
	/// - Parameters:
	///   - inputSize: Size of inputs of the layer for feed forward operations.
	///   - kernels: Convolution filters which are applied to inputs.
	///   - bias: Bias which is added onto the result of convolutions.
	///           There should be one bias value for each convolution kernel.
	///   - horizontalStride: Stride at which the input volume is traversed horizontally.
	///   - verticalStride: Stride at which the input volume is traversed vertically.
	///   - horizontalInset: Inset at which the horizontal traversion of the input volume begins.
	///   - verticalInset: Inset at which the vertical traversion of the input volume begins.
	public init(
		inputSize: (width: Int, height: Int, depth: Int),
		kernels: [Matrix3],
		bias: [Float],
		horizontalStride: Int,
		verticalStride: Int,
		horizontalInset: Int,
		verticalInset: Int
	)
	{
		self.inputSize = inputSize
		self.kernels = kernels
		self.bias = bias
		self.horizontalStride = horizontalStride
		self.verticalStride = verticalStride
		self.horizontalInset = horizontalInset
		self.verticalInset = verticalInset
		
		self.kernelGradients = kernels.map{$0.map{_ in 0}}
		self.biasGradients = bias.map{_ in 0}
	}
	
	
	public init(
		inputSize: (width: Int, height: Int, depth: Int),
		outputDepth: Int,
		kernelSize: (width: Int, height: Int),
		stride: (horizontal: Int, vertical: Int) = (1, 1),
		inset: (horizontal: Int, vertical: Int) = (0, 0)
	)
	{
		self.inputSize = inputSize
		self.horizontalStride = stride.horizontal
		self.verticalStride = stride.vertical
		self.horizontalInset = inset.horizontal
		self.verticalInset = inset.vertical
		self.bias = RandomWeightMatrix(width: outputDepth, height: 1).values
		self.kernels = (0 ..< outputDepth).map{_ in RandomWeightMatrix(width: kernelSize.width, height: kernelSize.height, depth: inputSize.depth)}
		
		self.kernelGradients = kernels.map{$0.map{_ in 0}}
		self.biasGradients = bias.map{_ in 0}
	}
	
	
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	public func forward(_ input: Matrix3) -> Matrix3
	{
		var output = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)

		for (z, kernel) in kernels.enumerated()
		{
			let convolved = input.convolved(
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
	///   - inputs: Inputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public mutating func updateGradients(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3) -> Matrix3
	{
		var gradient = Matrix3(repeating: 0, width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)

		for (outputZ, kernel) in kernels.enumerated()
		{
			var kernelGradient = kernelGradients[outputZ]
			var biasGradient: Float = biasGradients[outputZ]
			
			for outputY in 0 ..< outputSize.height
			{
				for outputX in 0 ..< outputSize.width
				{
					let outputGradient = nextLayerGradients[outputX, outputY, outputZ]
					
					kernelGradient += inputs[
						x: outputX * horizontalStride + horizontalInset,
						y: outputY * verticalStride + verticalInset,
						z: 0,
						width: kernel.width,
						height: kernel.height,
						depth: kernel.depth
					].mapv{$0 &* outputGradient}
					
					gradient[
						x: outputX * horizontalStride + horizontalInset,
						y: outputY * verticalStride + verticalInset,
						z: 0,
						width: kernel.width,
						height: kernel.height,
						depth: kernel.depth
					] += kernel.mapv{$0 &* outputGradient}
					
					biasGradient += outputGradient
				}
			}
			
			biasGradients[outputZ] = biasGradient
			kernelGradients[outputZ] = kernelGradient
		}
		
		return gradient
	}
	
}

/// A pooling layer for reducing dimensionality using max pooling.
public struct PoolingLayer: BidirectionalLayer
{
	
	/// Input size of the layer.
	/// Should not change after initialization
	public let inputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	public let outputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Creates a new Max Pooling layer with the given input and output size.
	/// Scaling factors are determined automatically from the input and output size.
	///
	/// A pooling layer can be used to reduce the size of the forwarded volume
	/// and thereby reducing the computational cost of subsequent layers.
	/// A pooling layer can also increase the size of the receptive field of subsequent convolution layers
	/// and prevent overfitting.
	///
	/// - Parameters:
	///   - inputSize: Size of the layer input
	///   - outputSize: Size of the layer output
	public init(inputSize: (width: Int, height: Int, depth: Int), outputSize: (width: Int, height: Int, depth: Int))
	{
		self.inputSize = inputSize
		self.outputSize = outputSize
	}
	
	
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	public func forward(_ input: Matrix3) -> Matrix3
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
			
			let submatrix = input[x: xOffset, y: yOffset, z: zOffset, width: xScale, height: yScale, depth: zScale]
			output[x, y, z] = max(submatrix.values)
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
	///   - inputs: Inputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public func updateGradients(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3) -> Matrix3
	{
		let xScale = inputSize.width / outputSize.width
		let yScale = inputSize.height / outputSize.height
		let zScale = inputSize.depth / outputSize.depth
		
		var gradients = Matrix3(repeating: 0, width: inputSize.width, height: inputSize.height, depth: inputSize.depth)

		for z in 0 ..< outputSize.depth
		{
			for y in 0 ..< outputSize.height
			{
				for x in 0 ..< outputSize.width
				{
					let inputVolume = inputs[
						x: x * xScale,
						y: y * yScale,
						z: z * zScale,
						width: xScale,
						height: yScale,
						depth: zScale
					]
					let (maxX, maxY, maxZ) = inputVolume.maxIndex()
					gradients[
						x * xScale + maxX,
						y * yScale + maxY,
						z * zScale + maxZ
					] = nextLayerGradients[x, y, z]
				}
			}
		}
		
		return gradients
	}
	
}


/// A layer which reshapes the output of one layer to fit the input of another layer
public struct ReshapingLayer: BidirectionalLayer
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
	
	
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	public func forward(_ input: Matrix3) -> Matrix3
	{
		return input.reshaped(width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
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
	///   - inputs: Inputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public func updateGradients(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3) -> Matrix3
	{
		return nextLayerGradients.reshaped(width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
	}
	
}


/// A layer which adds nonlinearity to forwarded data, which improves performance 
/// on nonlinear classification and regression tasks
public struct NonlinearityLayer: BidirectionalLayer, OutputLayer
{
	/// Output size of the layer.
	/// Should not change after initialization
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return inputSize
	}

	/// Input size of the layer.
	/// Should not change after initialization
	public let inputSize: (width: Int, height: Int, depth: Int)

	
	/// Activation function used for the layer
	public let activation: Activation
	
	
	/// Creates a new nonlinearity layer with a given nonlinearity function.
	///
	/// Nonlinearity is required for a network to perform nonlinear classification and regression.
	///
	/// - Parameters:
	///   - inputSize: Input size of the layer. The output size will be equal to the input size.
	///   - activation: Nonlinear activation function used by this layer.
	public init(inputSize: (width: Int, height: Int, depth: Int), activation: Activation)
	{
		self.inputSize = inputSize
		self.activation = activation
	}
	
	
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	public func forward(_ input: Matrix3) -> Matrix3
	{
		return input.mapv(activation.function)
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
	///   - inputs: Inputs of the current layer
	///   - learningRate: Learning rate at which the weights should be adjusted
	///   - momentum: Momentum of weight updates
	///   - decay: Decay rate at which weights should be decreased
	/// - Returns: Error matrix of the current layer
	public func updateGradients(nextLayerGradients: Matrix3, inputs: Matrix3, outputs: Matrix3) -> Matrix3
	{
		// gradients for anterior layer = gradients of posterior layer * derivative of gradients of posterior layer
		return Matrix3(
			values: activation.derivative(outputs.values) &* nextLayerGradients.values,
			width: inputSize.width,
			height: inputSize.height,
			depth: inputSize.depth
		)
	}
	
	public func loss(expected: Matrix3, output: Matrix3) -> Matrix3
	{
		if activation == .softmax
		{
			return Matrix3(
				values: output.values &- expected.values,
				width: inputSize.width,
				height: inputSize.height,
				depth: inputSize.depth
			)
		}
		else
		{
			return Matrix3(
				values: (expected.values &- output.values) &* activation.derivative(output.values),
				width: inputSize.width,
				height: inputSize.height,
				depth: inputSize.depth
			)
		}
	}
}
