//
//  GPUConvolutionLayer.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 11.04.17.
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
import Metal
import MatrixVector
import NeuralKit


/// A layer which is connected to the next layer using convolution kernels.
@available(OSX 10.12, *)
public struct GPUConvolutionLayer: GPUBidirectionalLayer, GPUWeightAdjustableLayer
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
	
	
	public var weights: [GPUTensor]
	{
		return [.vector(self.gpuBias, length: self.bias.count), .matrix3(self.gpuKernels)]
	}
	
	public var weightGradients: [GPUTensor]
	{
		return [.vector(self.gpuBiasGradient, length: self.bias.count), .matrix3(self.gpuKernelGradient)]
	}
	
	public var gradient: GPUMatrix3?
	{
		return gpuGradient
	}
	
	public var activation: GPUMatrix3?
	{
		return gpuOutput
	}
	
	private var gpuFunctionPipelineState: MTLComputePipelineState
	private var gpuBackpropagatePipelineState: MTLComputePipelineState
	private var gpuWeightGradientUpdatePipelineState: MTLComputePipelineState
	private var gpuBiasGradientUpdatePipelineState: MTLComputePipelineState
	
	private var gpuKernels: GPUMatrix3
	private var gpuBias: MTLBuffer
	
	private var gpuKernelGradient: GPUMatrix3
	private var gpuBiasGradient: MTLBuffer
	
	private var gpuOutput: GPUMatrix3
	private var gpuGradient: GPUMatrix3
	
	private var gpuHorizontalInset: MTLBuffer
	private var gpuVerticalInset: MTLBuffer
	private var gpuHorizontalStride: MTLBuffer
	private var gpuVerticalStride: MTLBuffer
	
	
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
		//		horizontalStride: Int, // strides other than 1 currently unsupported
		//		verticalStride: Int,
		horizontalInset: Int,
		verticalInset: Int
		)
	{
		self.inputSize = inputSize
		self.kernels = kernels
		self.bias = bias
		self.horizontalStride = 1
		self.verticalStride = 1
		self.horizontalInset = horizontalInset
		self.verticalInset = verticalInset
		
		
		guard
			let function = GPUGlobalLibrary.makeFunction(name: "ConvolutionLayer_forward"),
			let backpropagateFunction = GPUGlobalLibrary.makeFunction(name: "ConvolutionLayer_backpropagate"),
			let updateFunction = GPUGlobalLibrary.makeFunction(name: "ConvolutionLayer_update_gradients"),
			let updateBiasFunction = GPUGlobalLibrary.makeFunction(name: "ConvolutionLayer_update_bias_gradients")
			else
		{
			fatalError("Could not make Metal function.")
		}
		
		do
		{
			self.gpuFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: function)
			self.gpuBackpropagatePipelineState = try GPUGlobalDevice.makeComputePipelineState(function: backpropagateFunction)
			self.gpuWeightGradientUpdatePipelineState = try GPUGlobalDevice.makeComputePipelineState(function: updateFunction)
			self.gpuBiasGradientUpdatePipelineState = try GPUGlobalDevice.makeComputePipelineState(function: updateBiasFunction)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let allKernels = Matrix3(
			values: kernels.flatMap{$0.values},
			width:  kernels[0].width,
			height: kernels[0].height,
			depth:  kernels[0].depth * kernels.count
		)
		self.gpuKernels = GPUMatrix3(matrix: allKernels)
		self.gpuBias = GPUGlobalDevice.makeBuffer(
			bytes: bias,
			length: MemoryLayout.size(ofValue: Float(0)) * bias.count,
			options: []
			)!
		
		let kernelGradient = Matrix3(
			repeating: 0,
			width:  kernels[0].width,
			height: kernels[0].height,
			depth:  kernels[0].depth * kernels.count
		)
		self.gpuKernelGradient = GPUMatrix3(matrix: kernelGradient)
		
		self.gpuBiasGradient = GPUGlobalDevice.makeBuffer(
			bytes: (0..<kernels.count).map{_ in Float(0)},
			length: MemoryLayout<Float>.size * kernels.count,
			options: []
			)!
		
		
		let outputSize =
		{ () -> (width: Int, height: Int, depth: Int) in 
			let kernelWidth = kernels.first?.width ?? 0
			let kernelHeight = kernels.first?.height ?? 0
			
			let stridedWidth = inputSize.width
			let stridedHeight = inputSize.height
			
			return (
				width:  stridedWidth  - kernelWidth + 1 - (2 * horizontalInset),
				height: stridedHeight - kernelHeight + 1 - (2 * verticalInset),
				depth:  kernels.count
			)
		}()
		
		
		let outputValues = Matrix3(
			repeating: 0,
			width:  outputSize.width,
			height: outputSize.height,
			depth:  outputSize.depth
		)
		self.gpuOutput = GPUMatrix3(matrix: outputValues, isShared: false)
		
		let gradient = Matrix3(
			repeating: 0,
			width:  inputSize.width,
			height: inputSize.height,
			depth:  inputSize.depth
		)
		self.gpuGradient = GPUMatrix3(matrix: gradient)
		
		gpuHorizontalInset	= GPUGlobalDevice.makeBuffer(bytes: [Int32(self.horizontalInset)],	length: MemoryLayout<Int32>.size, options: [])!
		gpuVerticalInset	= GPUGlobalDevice.makeBuffer(bytes: [Int32(self.verticalInset)],		length: MemoryLayout<Int32>.size, options: [])!
		gpuHorizontalStride = GPUGlobalDevice.makeBuffer(bytes: [Int32(self.horizontalStride)],	length: MemoryLayout<Int32>.size, options: [])!
		gpuVerticalStride	= GPUGlobalDevice.makeBuffer(bytes: [Int32(self.verticalStride)],	length: MemoryLayout<Int32>.size, options: [])!
	}
	
	
	public init(
		inputSize: (width: Int, height: Int, depth: Int),
		outputDepth: Int,
		kernelSize: (width: Int, height: Int),
		inset: (horizontal: Int, vertical: Int) = (0, 0)
	)
	{
		let variance = 1 / Float(kernelSize.width * kernelSize.height * inputSize.depth + 1)
		
		let bias = RandomWeightMatrix(width: outputDepth, height: 1, variance: variance).values
		let kernels = (0 ..< outputDepth).map{_ in RandomWeightMatrix(width: kernelSize.width, height: kernelSize.height, depth: inputSize.depth, variance: variance)}
		
		self.init(inputSize: inputSize, kernels: kernels, bias: bias, horizontalInset: inset.horizontal, verticalInset: inset.vertical)
	}
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(self.gpuFunctionPipelineState)
		
		input.setBuffer(on: encoder,					   at: 0)
		gpuOutput.setBuffer(on: encoder,				   at: 2)
		gpuKernels.setBuffer(on: encoder,				   at: 4)
		
		encoder.setBuffer(gpuBias,				offset: 0, index: 6)
		
		encoder.setBuffer(gpuHorizontalInset,	offset: 0, index: 7)
		encoder.setBuffer(gpuVerticalInset,		offset: 0, index: 8)
		encoder.setBuffer(gpuHorizontalStride,	offset: 0, index: 9)
		encoder.setBuffer(gpuVerticalStride,	offset: 0, index: 10)
		
		encoder.dispatch(workSize: outputSize)
		
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
	public func backpropagate(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder
		) -> GPUMatrix3
	{
		encoder.setComputePipelineState(gpuBackpropagatePipelineState)
		
		inputs.setBuffer(on: encoder, at: 0)
		nextLayerGradients.setBuffer(on: encoder, at: 2)
		gpuGradient.setBuffer(on: encoder, at: 4)
		gpuKernels.setBuffer(on: encoder, at: 6)
		gpuKernelGradient.setBuffer(on: encoder, at: 8)
		
		encoder.setBuffer(gpuHorizontalInset,	offset: 0, index: 12)
		encoder.setBuffer(gpuVerticalInset,		offset: 0, index: 13)
		encoder.setBuffer(gpuHorizontalStride,	offset: 0, index: 14)
		encoder.setBuffer(gpuVerticalStride,	offset: 0, index: 15)
		
		encoder.dispatch(workSize: inputSize)
		
		encoder.setComputePipelineState(gpuWeightGradientUpdatePipelineState)
		encoder.dispatch(workSize: (width: kernels[0].width, height: kernels[0].height, depth: inputSize.depth * kernels.count))
		
		encoder.setComputePipelineState(gpuBiasGradientUpdatePipelineState)
		encoder.setBuffer(gpuBiasGradient, offset: 0, index: 8)
		encoder.dispatch(workSize: (width: outputSize.depth, height: 1, depth: 1))
		
		return gpuGradient
	}
	
	public mutating func finishTraining()
	{
		let updatedKernels = self.gpuKernels.asMatrix()
		let updatedBias = GPUMatrix(descriptor: (width: UInt32(kernels.count), height: 1), buffer: self.gpuBias).asMatrix().values
		
		self.bias = updatedBias
		
		for index in kernels.indices
		{
			kernels[index] = updatedKernels[
				x: 0,
				y: 0,
				z: inputSize.depth * index,
				width: kernels.first!.width,
				height: kernels.first!.height,
				depth: inputSize.depth
			]
		}
	}
}


@available(OSX 10.12, *)
extension GPUConvolutionLayer
{
	public init(_ layer: ConvolutionLayer)
	{
		self.init(
			inputSize: layer.inputSize,
			kernels: layer.kernels,
			bias: layer.bias,
			horizontalInset: layer.horizontalInset,
			verticalInset: layer.verticalInset
		)
	}
}
