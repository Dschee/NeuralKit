//
//  GPUReshapingLayer.swift
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
import NeuralKit

/// A layer which reshapes the output of one layer to fit the input of another layer
@available(OSX 10.12, *)
public struct GPUReshapingLayer: GPUBidirectionalLayer
{
	
	/// Output size of the layer.
	/// Should not change after initialization
	public let outputSize: (width: Int, height: Int, depth: Int)
	
	
	/// Input size of the layer.
	/// Should not change after initialization
	public let inputSize: (width: Int, height: Int, depth: Int)
	
	
	private var gpuOutputDescriptor: MTLBuffer
	
	
	public private(set) var gradient: GPUMatrix3?
	
	public var activation: GPUMatrix3?
	{
		return nil
	}
	
	
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
		
		gpuOutputDescriptor = GPUGlobalDevice.makeBuffer(
			bytes: [
				UInt32(outputSize.width),
				UInt32(outputSize.height),
				UInt32(outputSize.depth)
			],
			length: 3 * MemoryLayout<UInt32>.size,
			options: .storageModePrivate
		)
	}
	
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		return input.reshaped(
			descriptor: (
				width: UInt32(outputSize.width),
				height: UInt32(outputSize.height),
				depth: UInt32(outputSize.depth)
			),
			descriptorBuffer: gpuOutputDescriptor
		)
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
	public mutating func backpropagate(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder
		) -> GPUMatrix3
	{
		
		let gradient = nextLayerGradients.reshaped(
			descriptor: (
				width: UInt32(inputSize.width),
				height: UInt32(inputSize.height),
				depth: UInt32(inputSize.depth)
			),
			descriptorBuffer: inputs.descriptorBuffer
		)
		self.gradient = gradient
		return gradient
	}
	
}


@available(OSX 10.12, *)
public extension GPUReshapingLayer
{
	public init(_ layer: ReshapingLayer)
	{
		self.init(inputSize: layer.inputSize, outputSize: layer.outputSize)
	}
}
