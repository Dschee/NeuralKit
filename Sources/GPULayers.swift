//
//  GPULayers.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 04.04.17.
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

@available(OSX 10.12, *)
extension MTLComputeCommandEncoder
{
	func dispatch(workSize maxSize: (width: Int, height: Int, depth: Int))
	{
		let maxDeviceSize = self.device.maxThreadsPerThreadgroup
		
		var size = MTLSize(
			width: min(maxSize.width, maxDeviceSize.width),
			height: min(maxSize.height, maxDeviceSize.height),
			depth: min(maxSize.depth, maxDeviceSize.depth)
		)
		
		while size.width * size.height * size.depth > max(maxDeviceSize.width, maxDeviceSize.height, maxDeviceSize.depth)
		{
			// Shrink the largest size first, begin with depth
			// If there is no largest size, shrink anyway, begin with depth
			
			if size.depth > size.width && size.depth > size.height
			{
				size.depth /= 2
			}
			else if size.height > size.width && size.height > size.depth
			{
				size.height /= 2
			}
			else if size.width > size.height && size.width > size.depth
			{
				size.width /= 2
			}
			else if size.depth >= 2
			{
				size.depth /= 2
			}
			else if size.height >= 2
			{
				size.height /= 2
			}
			else if size.width >= 2
			{
				size.width /= 2
			}
			else
			{
				fatalError("Cannot make optimal size smaller. If you see this, there is something wrong.")
			}
		}
		
		let threadGroups = MTLSize(
			width:  (maxSize.width  + size.width  - 1) / size.width,
			height: (maxSize.height + size.height - 1) / size.height,
			depth:  (maxSize.depth  + size.depth  - 1) / size.depth
		)
		
		self.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: size)
	}
}


// A layer of a feed forward neural network
@available(OSX 10.12, *)
public protocol GPUNeuralLayer
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
	/// Performs data transformations for feed forward operation
	///
	/// - Parameter output: Input of the layer which should be forwarded
	/// - Returns: Weighted output of the layer
	func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
}

/// A layer of a feed forward neural network
@available(OSX 10.12, *)
public protocol GPUBidirectionalLayer: GPUNeuralLayer
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
	func backpropagate(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder
	) -> GPUMatrix3
	
	
	
}


@available(OSX 10.12, *)
public protocol GPUWeightAdjustableLayer: GPUNeuralLayer
{
	var weights: [GPUTensor] { get }
	var gradients: [GPUTensor] { get }
	
	mutating func finishTraining()
}


@available(OSX 10.12, *)
public protocol GPUOutputLayer: GPUNeuralLayer
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
