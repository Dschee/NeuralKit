//
//  GPUNeuralNetwork.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 04.04.17.
//
//

import Foundation
import Metal


/// A feed forward multi layer neural network
public struct GPUFeedForwardNeuralNetwork
{
	
	/// Neuron layers of the network
	public internal(set) var layers: [GPUNeuralLayer]
	
	public internal(set) var outputLayer: GPUOutputLayer
	
	public let device: MTLDevice
	private let queue: MTLCommandQueue
	
	
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
	public init?(layers: [GPUNeuralLayer], outputLayer: GPUOutputLayer)
	{
		guard (1..<layers.count)
			.map({layers[$0-1].outputSize == layers[$0].inputSize})
			.reduce(true, {$0 && $1})
			else
		{
			return nil
		}
		
		guard let device = MTLCreateSystemDefaultDevice() else
		{
			fatalError("Could not create default Metal device.")
		}
		self.device = device
		
		guard let library = try? device.makeLibrary(filepath: "/Users/Palle/Library/Developer/Xcode/DerivedData/NeuralKit-gqvgbrxdpkclopbfiglfaeqqojot/Build/Products/Release/NeuralKit.framework/Versions/A/Resources/default.metallib") else
		{
			fatalError("Could not create default Metal library.")
		}
		
		self.layers = layers
		self.outputLayer = outputLayer
		
		for index in layers.indices
		{
			self.layers[index].initialize(device: device, library: library)
		}
		
		self.outputLayer.initialize(device: device, library: library)
		
		self.queue = device.makeCommandQueue()
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
		let buffer = queue.makeCommandBuffer()
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
	
	
	/// Trains a network to match a given training sample more closely
	/// by backpropagating the error between the expected and actual value through the network.
	///
	/// The learning rate determines how fast the network should adapt.
	/// If it is chosen very small, the network will learn slower but also more accurately.
	///
	/// The returned error is calculated by summing the squares of the errors at each output neuron
	/// and multiplying it by 1/2.
	///
	/// - Parameters:
	///   - sample: Sample containing the input and expected value of the network
	///   - learningRate: Rate at which the network should adapt to the sample
	/// - Returns: The total error between the expected and actual output
	@discardableResult
	public mutating func train(_ sample: TrainingSample, learningRate: Float, annealingRate: Float = 0, momentum: Float = 0, decay: Float = 0) -> Float
	{
		fatalError()
		
		// Feed forward sample, keeping results of individual layers
		
//		var partialResults = Array<Matrix3>(repeating: Matrix3(repeating: 0, width: 0, height: 0, depth: 0), count: layers.count)
//		
//		var lastResult = sample.values
//		
//		for (index, layer) in layers.enumerated()
//		{
//			fatalError()
//			//			lastResult = layer.forward(lastResult)
//			//			partialResults[index] = lastResult
//		}
//		
//		let networkOutput = outputActivationFunction.function(lastResult.values)
//		
//		let errors: [Float]
//		
//		// Calculate the errors at the output layer
//		if outputActivationFunction == .softmax
//		{
//			errors = networkOutput &- sample.expected.values
//		}
//		else
//		{
//			errors = (networkOutput &- sample.expected.values) &* outputActivationFunction.derivative(networkOutput)
//		}
//		
//		let errorMatrix = Matrix3(
//			values: errors,
//			width: layers.last?.outputSize.width ?? 0,
//			height: layers.last?.outputSize.height ?? 0,
//			depth: layers.last?.outputSize.depth ?? 0
//		)
		
		
		// Backpropagate the error through the network
		
//		_ = layers.indices.reversed().reduce(errorMatrix)
//		{ (errorMatrix, layerIndex) -> Matrix3 in
//			layers[layerIndex].adjustWeights(
//				nextLayerGradients: errorMatrix,
//				inputs: layerIndex > 0 ? partialResults[layerIndex-1] : sample.values,
//				learningRate: learningRate,
//				annealingRate: annealingRate,
//				momentum: momentum,
//				decay: decay
//			)
//		}
		
		// Calculate the total error
//		return -sum(log(networkOutput) &* sample.expected.values)
	}
	
}
