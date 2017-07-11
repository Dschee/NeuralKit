//
//  FullyConnectedLayerTests.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.02.17.
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

import XCTest
@testable import NeuralKit
import MatrixVector

fileprivate let TestSetBase = "/Users/Palle/Developer/NeuralKit/Tests/NeuralKitTests/TestSets/"

class FullyConnectedLayerTests: XCTestCase
{
	var network: FeedForwardNeuralNetwork!
	
	func regressionSamples(from file: String) -> ([TrainingSample], [InputSample], [Matrix3])
	{
		guard let content = try? String(contentsOf: URL(fileURLWithPath: file + ".input.txt")) else { return ([],[],[]) }
		guard let outputContent = try? String(contentsOf: URL(fileURLWithPath: file + ".output.txt")) else { return ([],[],[]) }
		let lines = content.components(separatedBy: "\n")
		let trainingSampleRange:CountableRange<Int> = 0 ..< (lines.index(where: {$0.hasPrefix("0,0")}) ?? 1)
		let trainingSamples = trainingSampleRange.map
		{ line -> TrainingSample in
			let components = lines[line].components(separatedBy: ",").flatMap{Float($0)}
			let inputs = Array<Float>(components.dropLast())
			let output = components.last!
			let inputMatrix = Matrix3(values: inputs, width: 1, height: 1, depth: inputs.count)
			let outputMatrix = Matrix3(values: [output], width: 1, height: 1, depth: 1)
			return TrainingSample(values: inputMatrix, expected: outputMatrix)
		}
		let inputSampleRange = ((lines.index(where: {$0.hasPrefix("0,0")}) ?? 1) + 1) ..< lines.count
		let inputSamples = inputSampleRange.map
		{ line -> InputSample in
			let components = lines[line].components(separatedBy: ",").flatMap{Float($0)}
			let inputMatrix = Matrix3(values: components, width: 1, height: 1, depth: components.count)
			return InputSample(values: inputMatrix)
		}
		.filter{$0.values.indices.count > 0}
		
		let expectedOutputs = outputContent
			.components(separatedBy: "\n")
			.dropLast()
			.map {$0.components(separatedBy: ",")}
			.map {$0.flatMap{Float($0)}}
		
		let (normalizedTrainingInputs, scale: inputScale, offset: inputOffset) = TrainingSample.normalize(samples: trainingSamples.map{$0.values})
		let (normalizedTrainingOutputs, scale: outputScale, offset: outputOffset) = TrainingSample.normalize(samples: trainingSamples.map{$0.expected})
		
		let normalizedTrainingSamples = zip(normalizedTrainingInputs, normalizedTrainingOutputs).map{TrainingSample(values: $0, expected: $1)}
		let normalizedInputSamples = InputSample.normalize(samples: inputSamples.map{$0.values}, scale: inputScale, offset: inputOffset).map{InputSample(values: $0)}
		
		let normalizedExpectedOutputs = InputSample.normalize(samples: expectedOutputs.map{Matrix3.init(values: $0, width: 1, height: 1, depth: $0.count)}, scale: outputScale, offset: outputOffset)
		
		return (normalizedTrainingSamples, normalizedInputSamples, normalizedExpectedOutputs)
	}
	
	func binaryClassificationSamples(from file: String) -> ([TrainingSample], [InputSample], [Matrix3])
	{
		guard let content = try? String(contentsOf: URL(fileURLWithPath: file + ".input.txt")) else { return ([],[],[]) }
		guard let outputContent = try? String(contentsOf: URL(fileURLWithPath: file + ".output.txt")) else { return ([],[],[]) }
		let lines = content.components(separatedBy: "\n")
		let trainingSampleRange:CountableRange<Int> = 0 ..< (lines.index(where: {$0.hasPrefix("0,0,0")}) ?? 1)
		let inputSampleRange = ((lines.index(where: {$0.hasPrefix("0,0")}) ?? 1) + 1) ..< lines.count
		
		let trainingSamples = trainingSampleRange.map
		{ line -> TrainingSample in
			let components = lines[line].components(separatedBy: ",")
			let inputs = components.dropLast().flatMap{Float($0)}
			let result = Float(components.last!)!
			return TrainingSample(
				values: Matrix3(values: inputs, width: 1, height: 1, depth: inputs.count),
				expected: Matrix3(values: [result], width: 1, height: 1, depth: 1)
			)
		}
		
		let inputSampleValues = inputSampleRange
			.map {lines[$0]}
			.map {$0.components(separatedBy: ",")}
			.map {$0.flatMap{Float($0)}}
			.filter {$0.count > 0}
			
		let inputSamples = inputSampleValues
			.map {Matrix3(values: $0, width: 1, height: 1, depth: $0.count)}
			.map(InputSample.init)
		
		let (normalizedTrainingInputs, scale: inputScale, offset: inputOffset) = TrainingSample.normalize(samples: trainingSamples.map{$0.values})
		let normalizedInputSamples = InputSample.normalize(samples: inputSamples.map{$0.values}, scale: inputScale, offset: inputOffset).map{InputSample(values: $0)}
		let normalizedTrainingSamples = zip(normalizedTrainingInputs, trainingSamples.map{$0.expected}).map{TrainingSample(values: $0, expected: $1)}
		
		let expectedOutputs = outputContent
			.components(separatedBy: "\n")
			.dropLast()
			.map {$0.components(separatedBy: ",")}
			.map {$0.flatMap{Float($0)}}
			.map {Matrix3(values: $0, width: 1, height: 1, depth: $0.count)}
		
		return (normalizedTrainingSamples, normalizedInputSamples, expectedOutputs)
	}
	
    override func setUp()
	{
        super.setUp()
    }
    
    override func tearDown()
	{
        super.tearDown()
    }
	
	func performRegressionTest(network net: FeedForwardNeuralNetwork, trainingSamples: [TrainingSample], testSamples: [InputSample], expected: [Matrix3])
	{
		let session = NetworkTrainingSession(network: net, batchSize: 1, optimizer: SGDOptimizer(learningRate: 0.0004), normalizers: [], sampleProvider: ArrayTrainingSampleProvider(samples: trainingSamples))
		let sema = DispatchSemaphore(value: 0)
		session.onFinishTraining = { sema.signal() }
		session.train(epochs: 10_000)
		sema.wait()
		
		let network = session.network
		
		for (index, sample) in testSamples.enumerated()
		{
			let result = network.feedForward(sample.values)
			for (x,y,z) in expected[index].indices
			{
				print("ex: \(expected[index][x,y,z]), ac: \(result[x,y,z])")
				XCTAssertEqualWithAccuracy(result[x,y,z], expected[index][x,y,z], accuracy: 0.1)
			}
		}
	}
	
	
	func performBinaryClassificationTest(network net: FeedForwardNeuralNetwork, trainingSamples: [TrainingSample], testSamples: [InputSample], expected: [Matrix3])
	{
		let session = NetworkTrainingSession(network: net, batchSize: 1, optimizer: SGDOptimizer(learningRate: 0.003), normalizers: [], sampleProvider: ArrayTrainingSampleProvider(samples: trainingSamples))
		let sema = DispatchSemaphore(value: 0)
		session.onFinishTraining = { sema.signal() }
		session.train(epochs: 10_000)
		sema.wait()
		
		let network = session.network
		
		for (index, sample) in testSamples.enumerated()
		{
			let result = network.feedForward(sample.values)
			XCTAssertGreaterThan(result[0,0,0] * expected[index][0,0,0], 0) // If the sign matches, the classification is correct
		}
	}

	func testLinearRegression()
	{
		srand48(time(nil))
		let network = FeedForwardNeuralNetwork(
			layers: [
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 2, height: 1))
			],
			outputLayer: NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 1), activation: .linear)
		)!
		for x in ["a", "b", "c", "d"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = regressionSamples(from: TestSetBase + "Regression/Linear/\(x)")
			performRegressionTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
	func testNonlinearRegression()
	{
		srand48(time(nil))
		let	network = FeedForwardNeuralNetwork(
			layers: [
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 2, height: 4)),
				NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 4), activation: .tanh),
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 5, height: 1))
			],
			outputLayer: NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 1), activation: .linear)
		)!
		
		for x in ["a", "b", "c"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = regressionSamples(from: TestSetBase + "Regression/Nonlinear/\(x)")
			performRegressionTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
	func testLinearClassification()
	{
		srand48(time(nil))
		let network = FeedForwardNeuralNetwork(
			layers: [FullyConnectedLayer(weights: RandomWeightMatrix(width: 3, height: 1))],
			outputLayer: NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 1), activation: .tanh)
		)!
		for x in ["a", "b", "c"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = binaryClassificationSamples(from: TestSetBase + "Classification/Linear/\(x)")
			performBinaryClassificationTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
	func testNonlinearClassification()
	{
		srand48(time(nil))
		let network = FeedForwardNeuralNetwork(
			layers: [FullyConnectedLayer(weights: RandomWeightMatrix(width: 3, height: 6)),
			         FullyConnectedLayer(weights: RandomWeightMatrix(width: 7, height: 1))],
			outputLayer: NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 1), activation: .tanh)
		)!
		for x in ["a", "b", "c"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = binaryClassificationSamples(from: TestSetBase + "Classification/Nonlinear/\(x)")
			performBinaryClassificationTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
	func testLinearRegression2()
	{
		var net = FeedForwardNeuralNetwork(layers: [FullyConnectedLayer(inputDepth: 1, outputDepth: 1)], outputLayer: NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 1), activation: .linear))!
		
		let points: [[Float]] = [[2,2], [0,1], [1,3]]
		
		let samples = points.map { point -> TrainingSample in
			return TrainingSample(
				values: Matrix3(repeating: point[0], width: 1, height: 1, depth: 1),
				expected: Matrix3(repeating: point[1], width: 1, height: 1, depth: 1)
			)
		}
		
		let session = NetworkTrainingSession(network: net, batchSize: 1, optimizer: MomentumOptimizer(learningRate: 0.001, momentum: 0.6), normalizers: [L2Normalizer(decay: 0.0001)], sampleProvider: ArrayTrainingSampleProvider(samples: samples))
		
		let sema = DispatchSemaphore(value: 0)
		session.onFinishTraining = { sema.signal() }
		session.train(epochs: 100000)
		sema.wait()
		
		net = session.network
		let result = (net.layers[0] as! FullyConnectedLayer).weightMatrix
		print(result)
		
		XCTAssertEqualWithAccuracy(result[0, 0], 0.5, accuracy: 0.05)
		XCTAssertEqualWithAccuracy(result[1, 0], 1.5, accuracy: 0.05)
	}
	
}
