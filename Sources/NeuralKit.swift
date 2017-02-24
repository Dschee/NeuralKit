//
//  NeuralKit.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 19.02.17.
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
import Accelerate

public struct NeuralNetwork
{
	public internal(set) var layers: [NeuralLayer]
	
	public let outputActivationFunction: (([Float]) -> [Float])?
	public let outputActivationDerivative: (([Float]) -> [Float])?
	
	public init(layers: [NeuralLayer], outputActivation: (([Float]) -> [Float])? = nil, outputActivationDerivative: (([Float]) -> [Float])? = nil)
	{
		for i in 1 ..< layers.count
		{
			precondition(layers[i-1].outputSize == layers[i].inputSize, "Layers \(i-1) and \(i) must have matching output and input size")
		}
		self.layers = layers
		self.outputActivationFunction = outputActivation
		self.outputActivationDerivative = outputActivationDerivative
	}
	
	// Crashes the compiler
//	public init(layers: NeuralLayer..., outputActivation: (([Float]) -> [Float])? = nil, outputActivationDerivative: (([Float]) -> [Float])? = nil)
//	{
//		self.init(layers: layers, outputActivation: outputActivation, outputActivationDerivative: outputActivationDerivative)
//	}
	
	public func feedForward(_ sample: Matrix3) -> Matrix3
	{
		let lastLayerOutput = layers.reduce(sample)
		{
			sample, layer in
			layer.forward(sample)
		}
		return Matrix3(
			values: outputActivationFunction?(lastLayerOutput.values) ?? lastLayerOutput.values,
			width: lastLayerOutput.width,
			height: lastLayerOutput.height,
			depth: lastLayerOutput.depth
		)
	}
	
	@discardableResult
	public mutating func train(_ sample: TrainingSample, learningRate: Float) -> Float
	{
		var partialResults = Array<Matrix3>(repeating: Matrix3(repeating: 0, width: 0, height: 0, depth: 0), count: layers.count)
		var lastPartialResult:Matrix3 = sample.values
		var lastWeightedResult:Matrix3 = sample.values
		for (index, layer) in layers.enumerated()
		{
//			lastPartialResult = layer.forward(lastPartialResult)
			lastPartialResult = layer.activated(lastWeightedResult)
			lastWeightedResult = layer.weighted(lastPartialResult)
			partialResults[index] = lastPartialResult
		}
		let lastResult = outputActivationFunction?(lastWeightedResult.values) ?? lastWeightedResult.values
		
		let errors:[Float]
		if let outputActivationDerivative = self.outputActivationDerivative
		{
			errors = (sample.expected.values &- lastResult) &* outputActivationDerivative(lastResult)
		}
		else
		{
			errors = (sample.expected.values &- lastResult)
		}
		
		let errorMatrix = Matrix3(
			values: errors,
			width: layers.last?.outputSize.width ?? 0,
			height: layers.last?.outputSize.height ?? 0,
			depth: layers.last?.outputSize.depth ?? 0
		)
		
		_ = layers.indices.reversed().reduce(errorMatrix)
		{ (errorMatrix, layerIndex) -> Matrix3 in
			layers[layerIndex].adjustWeights(nextLayerErrors: errorMatrix, outputs: partialResults[layerIndex], learningRate: learningRate)
		}
		
		var totalError:Float = 0
		vDSP_svesq(errors, 1, &totalError, UInt(errors.count))
		return totalError
	}
	
}
