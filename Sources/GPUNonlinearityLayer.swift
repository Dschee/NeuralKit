//
//  GPUNonlinearityLayer.swift
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


/// A layer which adds nonlinearity to forwarded data, which improves performance
/// on nonlinear classification and regression tasks
public struct GPUNonlinearityLayer: GPUBidirectionalLayer, GPUOutputLayer
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
	
	private var gpuFunctionPipelineState: MTLComputePipelineState?
	private var gpuOutput: GPUMatrix3?
	
	private var gpuBackpropagateFunctionPipelineState: MTLComputePipelineState?
	private var gpuLossFunctionPipelineState: MTLComputePipelineState!
	private var gpuGradient: GPUMatrix3!
	
	public mutating func initialize(library: MTLLibrary, shareOutput: Bool)
	{
		let functionName: String?
		
		switch activation
		{
		case .linear:
			functionName = nil
			
		case .relu:
			functionName = "relu"
			
		case .sigmoid:
			functionName = "sigmoid"
			
		case .tanh:
			functionName = "tanh"
			
		case .softmax:
			fatalError("The softmax function is unavailable. Use SoftmaxLayer instead.")
		}
		
		if
			let funcName = functionName,
			let function = library.makeFunction(name: "NonlinearityLayer_forward_\(funcName)"),
			let backpropagationFunction = library.makeFunction(name: "NonlinearityLayer_backpropagate_\(funcName)")
		{
			do
			{
				self.gpuFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: function)
				self.gpuBackpropagateFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: backpropagationFunction)
			}
			catch
			{
				fatalError("\(error)")
			}
			
			let outputMatrix = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
			self.gpuOutput = GPUMatrix3(matrix: outputMatrix, isShared: shareOutput)
		}
		
		guard
			let lossFunction = library.makeFunction(name: "Loss_delta")
			else
		{
			fatalError()
		}
		
		do
		{
			self.gpuLossFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: lossFunction)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let gradientMatrix = Matrix3(repeating: 0, width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
		self.gpuGradient = GPUMatrix3(matrix: gradientMatrix)
	}
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		guard
			let gpuFunctionPipelineState = self.gpuFunctionPipelineState,
			let gpuOutput = self.gpuOutput
			else
		{
			return input
		}
		
		
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
		guard
			let gpuBackpropagateFunctionPipelineState = self.gpuBackpropagateFunctionPipelineState
			else
		{
			return nextLayerGradients
		}
		
		encoder.setComputePipelineState(gpuBackpropagateFunctionPipelineState)
		
		nextLayerGradients.setBuffer(on: encoder, at: 0)
		gpuGradient.setBuffer(on: encoder, at: 2)
		
		encoder.dispatch(workSize: outputSize)
		
		return gpuGradient
	}
	
	public func loss(expected: GPUMatrix3, actual: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(gpuLossFunctionPipelineState)
		
		expected.setBuffer(on: encoder, at: 0)
		actual.setBuffer(on: encoder, at: 2)
		gpuGradient.setBuffer(on: encoder, at: 4)
		
		encoder.dispatch(workSize: outputSize)
		
		guard
			let gpuBackpropagateFunctionPipelineState = self.gpuBackpropagateFunctionPipelineState
			else
		{
			return gpuGradient
		}
		
		encoder.setComputePipelineState(gpuBackpropagateFunctionPipelineState)
		
		gpuGradient.setBuffer(on: encoder, at: 0)
		gpuGradient.setBuffer(on: encoder, at: 2)
		
		encoder.dispatch(workSize: outputSize)
		
		return gpuGradient
	}
}

