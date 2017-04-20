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

// Extension which automatically computes the thread group size
// when dispatching a compute shader.
@available(OSX 10.12, *)
extension MTLComputeCommandEncoder
{
	
	/// Dispatches a compute shader with the given global size.
	///
	/// The size will be divided into thread groups are as large as possible
	/// for the device used.
	///
	/// This may lead to kernel invokations, which are outside of
	/// the given global size.
	///
	/// - Parameter maxSize: Global size
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


/// A layer of a feed forward neural network
/// which runs on a metal capable compute device.
///
/// A neural layer provides a forward propagation function
/// which takes an input, which can either be the network
/// input or the output of an anterior layer and performs
/// transformations on it using metal compute shaders.
///
/// A layer always has a fixed input and output size to ensure
/// that all layers will fit together in a network.
///
/// This protocol requires macOS 10.12 or greater.
@available(OSX 10.12, *)
public protocol GPUNeuralLayer
{
	
	/// Input size of the layer.
	/// Should not change after initialization
	var inputSize: (width: Int, height: Int, depth: Int) { get }
	
	
	/// Output size of the layer.
	/// Should not change after initialization
	var outputSize: (width: Int, height: Int, depth: Int) { get }
	
	
	/// Performs all transformations of the layer used in feed forward mode.
	///
	/// This function takes an input, which can either be the network
	/// input or the output of an anterior layer and performs
	/// transformations on it.
	///
	/// This function returns a matrix, which will contain the result of 
	/// the transformations after the shaders of this layer have finished
	/// running.
	///
	/// - Parameters:
	///   - input: Input of the layer.
	///   - encoder: Encoder used to dispatch metal compute shaders
	/// - Returns: A matrix which will contain the result of all computations of the layer.
	func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
}


/// A layer of a feed forward neural network, which supports backpropagation
/// of gradients.
/// 
/// During backproagation, a layer will be presented with the gradients
/// of the posterior layer and will compute the gradient, which can be passed to
/// the next layer.
/// If the layer has adjustable weights, the backproagation function will also
/// compute weight gradients, which can be used for weight updates by an optimizer.
@available(OSX 10.12, *)
public protocol GPUBidirectionalLayer: GPUNeuralLayer
{
	// Optional debug properties
	var gradient: GPUMatrix3? { get }
	var activation: GPUMatrix3? { get }
	
	
	/// Backpropagates the gradient through the network and calculates
	/// weight gradients if necessary.
	///
	/// The backpropagated gradient of this layer will be passed to the
	/// anterior layer.
	///
	/// - Parameters:
	///   - nextLayerGradients: Gradients computed by the posterior layer
	///   - inputs: Inputs of the layer
	///   - encoder: Encoder used to dispatch metal compute shaders
	/// - Returns: Gradients which will be passed to the anterior layer.
	mutating func backpropagate(
		nextLayerGradients: GPUMatrix3,
		inputs: GPUMatrix3,
		encoder: MTLComputeCommandEncoder
	) -> GPUMatrix3
}


/// A layer which has adjustable weights, which can be updated using backpropagation.
///
/// An adjustable layer will update its weight gradients during backpropagation.
/// An optimizer can then take the gradients and apply them to the weights of this layer.
@available(OSX 10.12, *)
public protocol GPUWeightAdjustableLayer: GPUBidirectionalLayer
{
	
	/// All weights of this layer.
	///
	/// The weight at a given index must correspond to the
	/// gradient at the same index
	var weights: [GPUTensor] { get }
	
	
	/// All weight gradients of this layer
	///
	/// The weight gradient at a given index must correspond to the
	/// weight at the same index.
	var weightGradients: [GPUTensor] { get }
	
	
	/// Notifies the layer that training has finished and 
	/// weights can be copied back to the CPU if needed.
	mutating func finishTraining()
}


/// A layer which can be used as the last layer in a neuronal network.
/// 
/// The last layer of a network is responsible for calculating the gradient
/// at the output of a network with respect to the expected output.
@available(OSX 10.12, *)
public protocol GPUOutputLayer: GPUNeuralLayer
{
	
	/// Calculates the loss at this layer for an expected output value
	///
	/// - Parameters:
	///   - expected: Expected output values of the layer
	///   - actual: Actual output values of the layer
	///   - encoder: Encoder for dispatching on the GPU
	/// - Returns: Gradient matrix
	func loss(expected: GPUMatrix3, actual: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	
}
