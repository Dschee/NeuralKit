//
//  GPUNeuralNetwork.swift
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


private class _BundleIdentifyingClass {}


@available(OSX 10.12, *)
public let GPUGlobalDevice = MTLCreateSystemDefaultDevice()!
@available(OSX 10.12, *)
public let GPUGlobalQueue = GPUGlobalDevice.makeCommandQueue()
@available(OSX 10.12, *)
public let GPUGlobalLibrary = try! GPUGlobalDevice.makeDefaultLibrary(bundle: Bundle(for: _BundleIdentifyingClass.self))


/// A feed forward multi layer neural network
@available(OSX 10.12, *)
public struct GPUFeedForwardNeuralNetwork
{
	
	/// Neuron layers of the network
	public internal(set) var layers: [GPUBidirectionalLayer]
	
	public internal(set) var outputLayer: GPUOutputLayer
	
	/// Creates a new neural network using the given layers and an activation function for the output layer.
	///
	/// The input size of all layers must match the output size of their anterior layers.
	///
	/// The initialization will fail if the input and output sizes of successive layers do not match.
	///
	/// - Parameters:
	///   - layers: Layers of the neural network.
	///   - outputActivation: Activation function which should be applied at the output or nil if a linear activation function should be used
	///   - outputActivationDerivative: Derivative of the output activation function or nil if a linear activation function should be used
	public init?(layers: [GPUBidirectionalLayer], outputLayer: GPUOutputLayer, library: MTLLibrary? = nil)
	{
		guard (1..<layers.count)
			.map({layers[$0-1].outputSize == layers[$0].inputSize})
			.reduce(true, {$0 && $1})
			else
		{
			return nil
		}
		
		self.layers = layers
		self.outputLayer = outputLayer
		
		
		
//		let lib = library ?? GPUGlobalDevice.newDefaultLibrary()!
	}
	
	
	/// Feeds a sample forward through the network.
	///
	/// This applies the activation function of each layer
	/// and passes the weighted output to the next layer until
	/// the last layer has been reached.
	///
	/// The weighted result of the forward operation on the last layer
	/// will be passed through the output activation function (if set) and will be returned.
	///
	/// - Parameter sample: Input of the network
	/// - Returns: Result of the feed forward operation on the last layer
	public func feedForward(_ sample: GPUMatrix3) -> Matrix3
	{
		let buffer = GPUGlobalQueue.makeCommandBuffer()
		let encoder = buffer.makeComputeCommandEncoder()
		
		let lastHiddenLayerResult = layers.reduce(sample)
		{ layerInput, layer in
			let layerResult = layer.forward(layerInput, encoder: encoder)
			return layerResult
		}
		
		let output = outputLayer.forward(lastHiddenLayerResult, encoder: encoder)
		
		encoder.endEncoding()
		buffer.commit()
		buffer.waitUntilCompleted()
		
		return output.asMatrix()
	}
	
	
	/// Updates the gradients of all layers of the network using the 
	/// specified input and expected output.
	///
	/// - Parameters:
	///   - sample: Sample for which the gradient should be calculated.
	///   - encoder: Encoder for metal kernel dispatch
	internal mutating func updateGradients(with sample: (input: GPUMatrix3, expected: GPUMatrix3), encoder: MTLComputeCommandEncoder)
	{
		var layerInputs = [sample.input]
		
		for layer in self.layers
		{
			layerInputs.append(layer.forward(layerInputs.last!, encoder: encoder))
		}
		
		layerInputs.append(outputLayer.forward(layerInputs.last!, encoder: encoder))
		
		var gradient = outputLayer.loss(expected: sample.expected, actual: layerInputs.last!, encoder: encoder)
		
		for index in layers.indices.reversed()
		{
			gradient = self.layers[index].backpropagate(
				nextLayerGradients: gradient,
				inputs: layerInputs[index],
				encoder: encoder
			)
		}
	}
	
	
	/// Finishes training by copying back all weights from the GPU to the CPU.
	mutating func finishTraining()
	{
		for (index, layer) in layers.enumerated() where layer is GPUWeightAdjustableLayer
		{
			var l = layer as! GPUWeightAdjustableLayer & GPUBidirectionalLayer
			l.finishTraining()
			layers[index] = l
		}
	}
}
