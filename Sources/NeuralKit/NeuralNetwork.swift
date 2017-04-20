//
//  NeuralKit.swift
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



/// A feed forward multi layer neural network
public struct FeedForwardNeuralNetwork
{
	
	/// Neuron layers of the network
	public internal(set) var layers: [BidirectionalLayer]
	
	
	/// Activation function at the output layer
	public internal(set) var outputLayer: OutputLayer
	
	
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
	public init?(layers: [BidirectionalLayer], outputLayer: OutputLayer)
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
	public func feedForward(_ sample: Matrix3) -> Matrix3
	{
		let lastLayerOutput = layers.reduce(sample)
		{
			forwarded, layer in
			layer.forward(forwarded)
		}
		return outputLayer.forward(lastLayerOutput)
	}
	
	
	public mutating func backpropagate(_ sample: TrainingSample)
	{
		var layerOutputs = [sample.values]
		
		for layer in layers
		{
			layerOutputs.append(layer.forward(layerOutputs.last!))
		}
		
		layerOutputs.append(outputLayer.forward(layerOutputs.last!))
		
		var loss = outputLayer.loss(expected: sample.expected, output: layerOutputs.last!)
		
		for index in layers.indices.reversed()
		{
			loss = layers[index].updateGradients(nextLayerGradients: loss, inputs: layerOutputs[index], outputs: layerOutputs[index+1])
		}
	}
	
}
