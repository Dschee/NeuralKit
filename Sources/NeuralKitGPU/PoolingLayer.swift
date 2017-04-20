//
//  GPUPoolingLayer.swift
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


/// A pooling layer for reducing dimensionality using max pooling.
@available(OSX 10.12, *)
public struct GPUPoolingLayer: GPUBidirectionalLayer
{
	
	/// Input size of the layer.
	/// Should not change after initialization
	public let inputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	public let outputSize: (width: Int, height: Int, depth: Int)
	
	
	private var gpuFunctionPipelineState: MTLComputePipelineState
	private var gpuOutput: GPUMatrix3
	
	private var gpuBackpropagateFunctionPipelineState: MTLComputePipelineState
	private var gpuGradient: GPUMatrix3
	
	public var gradient: GPUMatrix3?
	{
		return gpuGradient
	}
	
	public var activation: GPUMatrix3?
	{
		return gpuOutput
	}
	
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
		
		guard
			let function = GPUGlobalLibrary.makeFunction(name: "PoolingLayer_forward"),
			let backpropagateFunction = GPUGlobalLibrary.makeFunction(name: "PoolingLayer_backpropagate")
			else
		{
			fatalError("Could not make Metal function.")
		}
		
		do
		{
			self.gpuFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: function)
			self.gpuBackpropagateFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: backpropagateFunction)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let outputValues = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		self.gpuOutput = GPUMatrix3(matrix: outputValues, isShared: false)
		
		let gradients = Matrix3(repeating: 0, width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
		self.gpuGradient = GPUMatrix3(matrix: gradients)
	}
	
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(gpuFunctionPipelineState)
		
		input.setBuffer(on: encoder, at: 0)
		gpuOutput.setBuffer(on: encoder, at: 2)
		
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
		encoder.setComputePipelineState(gpuBackpropagateFunctionPipelineState)
		
		inputs.setBuffer(on: encoder, at: 0)
		nextLayerGradients.setBuffer(on: encoder, at: 2)
		gpuGradient.setBuffer(on: encoder, at: 4)
		
		encoder.dispatch(workSize: outputSize)
		
		return gpuGradient
	}
	
}


@available(OSX 10.12, *)
public extension GPUPoolingLayer
{
	public init(_ layer: PoolingLayer)
	{
		self.init(inputSize: layer.inputSize, outputSize: layer.outputSize)
	}
}
