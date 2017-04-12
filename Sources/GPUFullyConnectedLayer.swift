//
//  GPUFullyConnectedLayer.swift
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


/// A fully connected layer of a neural network.
/// All neurons of the layer are connected to all neurons of the next layer
public struct GPUFullyConnectedLayer: GPUBidirectionalLayer, GPUWeightAdjustableLayer
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
	
	public var weights: [GPUTensor]
	{
		return [.matrix(self.gpuWeights)]
	}
	
	public var gradients: [GPUTensor]
	{
		return [.matrix(self.gpuWeightGradient)]
	}
	
	
	private var gpuWeights: GPUMatrix
	private var gpuOutput: GPUMatrix3
	private var gpuFunctionPipelineState: MTLComputePipelineState
	
	private var gpuBackpropagateFunctionPipelineState: MTLComputePipelineState
	private var gpuGradientUpdateFunctionPipelineState: MTLComputePipelineState
	
	private var gpuGradient: GPUMatrix3
	private var gpuWeightGradient: GPUMatrix
	
	
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
		
		guard
			let function = GPUGlobalLibrary.makeFunction(name: "FullyConnectedLayer_forward"),
			let backpropagateFunction = GPUGlobalLibrary.makeFunction(name: "FullyConnectedLayer_backpropagate"),
			let gradientUpdateFunction = GPUGlobalLibrary.makeFunction(name: "FullyConnectedLayer_update_gradients")
			else
		{
			fatalError("Could not make Metal function.")
		}
		
		do
		{
			self.gpuFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: function)
			self.gpuBackpropagateFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: backpropagateFunction)
			self.gpuGradientUpdateFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: gradientUpdateFunction)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		self.gpuWeights = GPUMatrix(matrix: self.weightMatrix)
		
		let weightGradient = Matrix.init(repeating: 0, width: self.weightMatrix.width, height: self.weightMatrix.height)
		self.gpuWeightGradient = GPUMatrix(matrix: weightGradient)
		
		let outputValues = Matrix3(repeating: 0, width: 1, height: 1, depth: weights.height)
		self.gpuOutput = GPUMatrix3(matrix: outputValues, isShared: false)
		
		let gradientValues = Matrix3(repeating: 0, width: 1, height: 1, depth: weights.width)
		self.gpuGradient = GPUMatrix3(matrix: gradientValues)
	}
	
	
	public init(inputDepth: Int, outputDepth: Int)
	{
		let weights = RandomWeightMatrix(width: inputDepth + 1, height: outputDepth, variance: 1 / Float(inputDepth + 1))
		self.init(weights: weights)
	}
	
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(self.gpuFunctionPipelineState)
		
		input.setBuffer(on: encoder, at: 0)
		gpuOutput.setBuffer(on: encoder, at: 2)
		gpuWeights.setBuffer(on: encoder, at: 4)
		
		encoder.dispatch(workSize: (width: outputSize.depth, height: 1, depth: 1))
		
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
		encoder.setComputePipelineState(self.gpuBackpropagateFunctionPipelineState)
		
		inputs.setBuffer(on: encoder, at: 0)
		nextLayerGradients.setBuffer(on: encoder, at: 2)
		gpuGradient.setBuffer(on: encoder, at: 4)
		gpuWeights.setBuffer(on: encoder, at: 6)
		gpuWeightGradient.setBuffer(on: encoder, at: 8)
		
		encoder.dispatch(workSize: (width: inputSize.depth, height: 1, depth: 1))
		
		encoder.setComputePipelineState(self.gpuGradientUpdateFunctionPipelineState)
		encoder.dispatch(workSize: (width: inputSize.depth + 1, height: outputSize.depth, depth: 1))
		
		return gpuGradient
	}
	
}

