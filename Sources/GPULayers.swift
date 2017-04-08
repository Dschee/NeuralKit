//
//  GPULayers.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 04.04.17.
//
//

import Foundation
import Metal

// A layer of a feed forward neural network
public protocol GPUBasicLayer
{
	
	/// Input size of the layer.
	/// Should not change after initialization
	var inputSize: (width: Int, height: Int, depth: Int) { get }
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	var outputSize: (width: Int, height: Int, depth: Int) { get }
	
	
	
	/// Initializes all buffers required for the layer to operate.
	///
	/// - Parameter device: Compute device, which is used.
	mutating func initialize(device: MTLDevice, library: MTLLibrary)
	
	
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3

}

/// A layer of a feed forward neural network
public protocol GPUNeuralLayer: GPUBasicLayer
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
	mutating func adjustWeights(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder,
		learningRate: Float,
		momentum: Float,
		decay: Float
	) -> GPUMatrix3
	
}


public protocol GPUOutputLayer: GPUBasicLayer
{
	
	/// Calculates the loss at this layer for an expected output value
	///
	/// - Parameters:
	///   - expected: Expected output values of the layer
	///   - actual: Actual output values of the layer
	///   - encoder: Encoder for dispatching on the GPU
	/// - Returns: Loss matrix
	func loss(expected: GPUMatrix3, actual: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	
}


/// A fully connected layer of a neural network.
/// All neurons of the layer are connected to all neurons of the next layer
public struct GPUFullyConnectedLayer: GPUNeuralLayer
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
		self.weights = weights
		self.weightDelta = Matrix(repeating: 0, width: weights.width, height: weights.height)
	}
	
	
	public init(inputDepth: Int, outputDepth: Int)
	{
		self.weights = RandomWeightMatrix(width: inputDepth + 1, height: outputDepth)
		self.weightDelta = Matrix(repeating: 0, width: weights.width, height: weights.height)
	}
	
	
	private var gpuWeights: GPUMatrix!
	private var gpuOutput: GPUMatrix3!
	private var gpuFunctionPipelineState: MTLComputePipelineState!
	
	private var gpuGradient: GPUMatrix3!
	private var gpuWeightDelta: GPUMatrix!
	private var gpuBackpropagateFunctionPipelineState: MTLComputePipelineState!
	
	public mutating func initialize(device: MTLDevice, library: MTLLibrary)
	{
		guard
			let function = library.makeFunction(name: "FullyConnectedLayer_forward"),
			let backpropagateFunction = library.makeFunction(name: "FullyConnectedLayer_backpropagate")
		else
		{
			fatalError("Could not make Metal function.")
		}
		
		do
		{
			self.gpuFunctionPipelineState = try device.makeComputePipelineState(function: function)
			self.gpuBackpropagateFunctionPipelineState = try device.makeComputePipelineState(function: backpropagateFunction)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		self.gpuWeights = GPUMatrix(on: device, matrix: self.weights)
		self.gpuWeightDelta = GPUMatrix(on: device, matrix: self.weightDelta)
		
		let outputValues = Matrix3(repeating: 0, width: self.outputSize.width, height: self.outputSize.height, depth: self.outputSize.depth)
		self.gpuOutput = GPUMatrix3(on: device, matrix: outputValues)
		
		let gradientValues = Matrix3(repeating: 0, width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
		self.gpuGradient = GPUMatrix3(on: device, matrix: gradientValues)
	}
	
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(self.gpuFunctionPipelineState)
		
		input.setBuffer(on: encoder, at: 0)
		gpuOutput.setBuffer(on: encoder, at: 2)
		gpuWeights.setBuffer(on: encoder, at: 4)
		
		let threadGroupSize = MTLSize(
			width: min(outputSize.depth, encoder.device.maxThreadsPerThreadgroup.width),
			height: 1,
			depth: 1
		)
		let threadGroups = MTLSize(width: (outputSize.depth + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
		
		encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
		
		return gpuOutput
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
	public mutating func adjustWeights(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder,
		learningRate: Float,
		momentum: Float,
		decay: Float
		) -> GPUMatrix3
	{
		encoder.setComputePipelineState(self.gpuBackpropagateFunctionPipelineState)
		
		inputs.setBuffer(on: encoder, at: 0)
		nextLayerGradients.setBuffer(on: encoder, at: 2)
		gpuGradient.setBuffer(on: encoder, at: 4)
		gpuWeights.setBuffer(on: encoder, at: 6)
		gpuWeightDelta.setBuffer(on: encoder, at: 8)
		
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 10)
		encoder.setBytes([momentum], length: MemoryLayout<Float>.size, at: 11)
		encoder.setBytes([decay], length: MemoryLayout<Float>.size, at: 12)
		
		let threadGroupSize = MTLSize(
			width: min(inputSize.depth, encoder.device.maxThreadsPerThreadgroup.width),
			height: 1,
			depth: 1
		)
		let threadGroups = MTLSize(width: (inputSize.depth + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
		
		encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
		
		return gpuGradient
	}
	
}


/// A layer which is connected to the next layer using convolution kernels.
public struct GPUConvolutionLayer: GPUNeuralLayer
{
	
	/// The convolution kernels which are applied to input values
	public private(set) var kernels: [Matrix3]
	
	
	/// The bias values which are applied after convolutions
	public private(set) var bias: [Float]
	
	
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
	}
	
	
	public init(
		inputSize: (width: Int, height: Int, depth: Int),
		outputDepth: Int,
		kernelSize: (width: Int, height: Int),
		//		stride: (horizontal: Int, vertical: Int) = (1, 1), // strides other than 1 currently unsupported
		inset: (horizontal: Int, vertical: Int) = (0, 0)
		)
	{
		self.inputSize = inputSize
		self.horizontalStride = 1
		self.verticalStride = 1
		self.horizontalInset = inset.horizontal
		self.verticalInset = inset.vertical
		self.bias = RandomWeightMatrix(width: outputDepth, height: 1).values
		self.kernels = (0 ..< outputDepth).map{_ in RandomWeightMatrix(width: kernelSize.width, height: kernelSize.height, depth: inputSize.depth)}
	}
	
	private var gpuFunctionPipelineState: MTLComputePipelineState!
	private var gpuKernels: GPUMatrix3!
	private var gpuBias: MTLBuffer!
	private var gpuOutput: GPUMatrix3!
	private var gpuHorizontalInset: MTLBuffer!
	private var gpuVerticalInset: MTLBuffer!
	private var gpuHorizontalStride: MTLBuffer!
	private var gpuVerticalStride: MTLBuffer!
	
	public mutating func initialize(device: MTLDevice, library: MTLLibrary)
	{
		let constantValues = MTLFunctionConstantValues()
		guard let function = try? library.makeFunction(name: "ConvolutionLayer_forward", constantValues: constantValues) else
		{
			fatalError("Could not make Metal function.")
		}
		
		do
		{
			self.gpuFunctionPipelineState = try device.makeComputePipelineState(function: function)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let allKernels = Matrix3(
			values: kernels.flatMap{$0.values},
			width: kernels[0].width,
			height: kernels[0].height,
			depth: kernels[0].depth * kernels.count
		)
		self.gpuKernels = GPUMatrix3(on: device, matrix: allKernels)
		self.gpuBias = device.makeBuffer(bytes: bias, length: MemoryLayout.size(ofValue: Float(0)) * bias.count, options: [])
		
		let outputValues = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		self.gpuOutput = GPUMatrix3(on: device, matrix: outputValues)
		
		gpuHorizontalInset	= device.makeBuffer(bytes: [Int32(self.horizontalInset)],	length: MemoryLayout<Int32>.size, options: [])
		gpuVerticalInset	= device.makeBuffer(bytes: [Int32(self.verticalInset)],		length: MemoryLayout<Int32>.size, options: [])
		gpuHorizontalStride = device.makeBuffer(bytes: [Int32(self.horizontalStride)],	length: MemoryLayout<Int32>.size, options: [])
		gpuVerticalStride	= device.makeBuffer(bytes: [Int32(self.verticalStride)],	length: MemoryLayout<Int32>.size, options: [])
	}
	
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(self.gpuFunctionPipelineState)
		
		input.setBuffer(on: encoder,					   at: 0)
		gpuOutput.setBuffer(on: encoder,				   at: 2)
		gpuKernels.setBuffer(on: encoder,				   at: 4)
		
		encoder.setBuffer(gpuBias,				offset: 0, at: 6)
		encoder.setBuffer(gpuHorizontalInset,	offset: 0, at: 7)
		encoder.setBuffer(gpuVerticalInset,		offset: 0, at: 8)
		encoder.setBuffer(gpuHorizontalStride,	offset: 0, at: 9)
		encoder.setBuffer(gpuVerticalStride,	offset: 0, at: 10)
		
		let threadGroupSize = MTLSize(
			width: min(encoder.device.maxThreadsPerThreadgroup.width, outputSize.width),
			height: min(encoder.device.maxThreadsPerThreadgroup.height, outputSize.height),
			depth: min(encoder.device.maxThreadsPerThreadgroup.depth, outputSize.depth)
		)
		let threadGroups = MTLSize(
			width: (outputSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
			height: (outputSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
			depth: (outputSize.depth + threadGroupSize.depth - 1) / threadGroupSize.depth
		)
		
		encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
		
		return gpuOutput
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
	public mutating func adjustWeights(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder,
		learningRate: Float,
		momentum: Float,
		decay: Float
		) -> GPUMatrix3
	{
		fatalError()
		
//		var gradient = Matrix3(repeating: 0, width: self.inputSize.width, height: self.inputSize.height, depth: self.inputSize.depth)
//		
//		for (outputZ, kernel) in kernels.enumerated()
//		{
//			var kernelGradient = Matrix3.init(repeating: 0, width: kernel.width, height: kernel.height, depth: kernel.depth)
//			var biasGradient: Float = 0
//			
//			for outputY in 0 ..< outputSize.height
//			{
//				for outputX in 0 ..< outputSize.width
//				{
//					let outputGradient = nextLayerGradients[outputX, outputY, outputZ]
//					
//					kernelGradient += inputs[
//						x: outputX * horizontalStride + horizontalInset,
//						y: outputY * verticalStride + verticalInset,
//						z: 0,
//						width: kernel.width,
//						height: kernel.height,
//						depth: kernel.depth
//						].mapv{$0 &* outputGradient}
//					
//					gradient[
//						x: outputX * horizontalStride + horizontalInset,
//						y: outputY * verticalStride + verticalInset,
//						z: 0,
//						width: kernel.width,
//						height: kernel.height,
//						depth: kernel.depth
//						] += kernel.mapv{$0 &* outputGradient}
//					
//					biasGradient += outputGradient
//				}
//			}
//			kernels[outputZ] += kernelGradient.mapv{$0 &* learningRate}
//			bias[outputZ] += biasGradient * learningRate
//		}
//		
//		return gradient
	}
	
}

/// A pooling layer for reducing dimensionality using max pooling.
public struct GPUPoolingLayer: GPUNeuralLayer
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
	
	
	private var gpuFunctionPipelineState: MTLComputePipelineState!
	private var gpuOutput: GPUMatrix3!
	
	public mutating func initialize(device: MTLDevice, library: MTLLibrary)
	{
		guard let function = library.makeFunction(name: "ConvolutionLayer_forward") else
		{
			fatalError("Could not make Metal function.")
		}
		
		do
		{
			self.gpuFunctionPipelineState = try device.makeComputePipelineState(function: function)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let outputValues = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		self.gpuOutput = GPUMatrix3(on: device, matrix: outputValues)
	}
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(gpuFunctionPipelineState)
		
		input.setBuffer(on: encoder, at: 0)
		gpuOutput.setBuffer(on: encoder, at: 2)
		
		let threadGroupSize = MTLSize(
			width: min(encoder.device.maxThreadsPerThreadgroup.width, outputSize.width),
			height: min(encoder.device.maxThreadsPerThreadgroup.height, outputSize.height),
			depth: min(encoder.device.maxThreadsPerThreadgroup.depth, outputSize.depth)
		)
		let threadGroups = MTLSize(
			width: (outputSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
			height: (outputSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
			depth: (outputSize.depth + threadGroupSize.depth - 1) / threadGroupSize.depth
		)
		
		encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
		
		return gpuOutput
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
	public mutating func adjustWeights(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder,
		learningRate: Float,
		momentum: Float,
		decay: Float
		) -> GPUMatrix3
	{
		fatalError()
//		let xScale = inputSize.width / outputSize.width
//		let yScale = inputSize.height / outputSize.height
//		let zScale = inputSize.depth / outputSize.depth
//		
//		var gradients = Matrix3(repeating: 0, width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
//		
//		for z in 0 ..< outputSize.depth
//		{
//			for y in 0 ..< outputSize.height
//			{
//				for x in 0 ..< outputSize.width
//				{
//					let inputVolume = inputs[
//						x: x * xScale,
//						y: y * yScale,
//						z: z * zScale,
//						width: xScale,
//						height: yScale,
//						depth: zScale
//					]
//					let (maxX, maxY, maxZ) = inputVolume.maxIndex()
//					gradients[
//						x * xScale + maxX,
//						y * yScale + maxY,
//						z * zScale + maxZ
//						] = nextLayerGradients[x, y, z]
//				}
//			}
//		}
//		
//		return gradients
	}
	
}


/// A layer which reshapes the output of one layer to fit the input of another layer
public struct GPUReshapingLayer: GPUNeuralLayer
{
	
	/// Output size of the layer.
	/// Should not change after initialization
	public let outputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Input size of the layer.
	/// Should not change after initialization
	public let inputSize: (width: Int, height: Int, depth: Int)
	
	
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
	
	private var gpuOutputDescriptor: MTLBuffer!
	
	public mutating func initialize(device: MTLDevice, library: MTLLibrary)
	{
		gpuOutputDescriptor = device.makeBuffer(bytes: [UInt32(outputSize.width), UInt32(outputSize.height), UInt32(outputSize.depth)], length: 3 * MemoryLayout<UInt32>.size, options: [])
	}
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		return input.reshaped(descriptor: (width: UInt32(outputSize.width), height: UInt32(outputSize.height), depth: UInt32(outputSize.depth)), descriptorBuffer: gpuOutputDescriptor)
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
	public mutating func adjustWeights(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder,
		learningRate: Float,
		momentum: Float,
		decay: Float
		) -> GPUMatrix3
	{
		fatalError()
//		return nextLayerGradients.reshaped(width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
	}
	
}


/// A layer which adds nonlinearity to forwarded data, which improves performance
/// on nonlinear classification and regression tasks
public struct GPUNonlinearityLayer: GPUNeuralLayer, GPUOutputLayer
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
		
		if activation == .linear
		{
			print("Warning: Using a linear function on a nonlinearity layer has no effect.")
		}
	}
	
	private var gpuFunctionPipelineState: MTLComputePipelineState!
	private var gpuOutput: GPUMatrix3!
	
	private var gpuBackpropagateFunctionPipelineState: MTLComputePipelineState!
	private var gpuGradient: GPUMatrix3!
	
	public mutating func initialize(device: MTLDevice, library: MTLLibrary)
	{
		let functionName: String
		
		switch activation
		{
		case .linear:
			return
		
		case .relu:
			functionName = "relu"
			
		case .sigmoid:
			functionName = "sigmoid"
			
		case .tanh:
			functionName = "tanh"
			
		case .softmax:
			fatalError("The softmax function is unavailable.")
		}
		
		guard
			let function = library.makeFunction(name: "NonlinearityLayer_forward_\(functionName)"),
			let backpropagationFunction = library.makeFunction(name: "NonlinearityLayer_backpropagate_\(functionName)")
		else
		{
			fatalError()
		}
		
		do
		{
			self.gpuFunctionPipelineState = try device.makeComputePipelineState(function: function)
			self.gpuBackpropagateFunctionPipelineState = try device.makeComputePipelineState(function: backpropagationFunction)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let outputMatrix = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		self.gpuOutput = GPUMatrix3(on: device, matrix: outputMatrix)
		
		let gradientMatrix = Matrix3(repeating: 0, width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
		self.gpuGradient = GPUMatrix3(on: device, matrix: gradientMatrix)
	}
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		guard activation != .linear else
		{
			return input
		}
		
		encoder.setComputePipelineState(self.gpuFunctionPipelineState)
		
		input.setBuffer(on: encoder, at: 0)
		gpuOutput.setBuffer(on: encoder, at: 2)
		
		let threadGroupSize = MTLSize(
			width: min(encoder.device.maxThreadsPerThreadgroup.width, outputSize.width),
			height: min(encoder.device.maxThreadsPerThreadgroup.height, outputSize.height),
			depth: min(encoder.device.maxThreadsPerThreadgroup.depth, outputSize.depth)
		)
		let threadGroups = MTLSize(
			width: (outputSize.width + threadGroupSize.width - 1) / threadGroupSize.width,
			height: (outputSize.height + threadGroupSize.height - 1) / threadGroupSize.height,
			depth: (outputSize.depth + threadGroupSize.depth - 1) / threadGroupSize.depth
		)
		
		encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
		
		return gpuOutput
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
	public mutating func adjustWeights(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder,
		learningRate: Float,
		momentum: Float,
		decay: Float
		) -> GPUMatrix3
	{
		fatalError()
		// gradients for anterior layer = gradients of posterior layer * derivative of gradients of posterior layer
//		return Matrix3(
//			values: activation.derivative(nextLayerGradients.values) &* nextLayerGradients.values,
//			width: inputSize.width,
//			height: inputSize.height,
//			depth: inputSize.depth
//		)
	}
	
	public func loss(expected: GPUMatrix3, actual: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		fatalError()
	}
}


public struct GPUSoftmaxLayer: GPUOutputLayer
{
	public let inputSize: (width: Int, height: Int, depth: Int)
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return inputSize
	}
	
	private var gpuFunctionPipelineState: MTLComputePipelineState!
	private var gpuOutput: GPUMatrix3!
	
	public mutating func initialize(device: MTLDevice, library: MTLLibrary)
	{
		guard let function = library.makeFunction(name: "SoftmaxLayer_forward") else
		{
			fatalError()
		}
		
		do
		{
			self.gpuFunctionPipelineState = try device.makeComputePipelineState(function: function)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let outputMatrix = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		self.gpuOutput = GPUMatrix3(on: device, matrix: outputMatrix)
	}
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		fatalError()
	}
	
	public func loss(expected: GPUMatrix3, actual: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		fatalError()
	}
}


