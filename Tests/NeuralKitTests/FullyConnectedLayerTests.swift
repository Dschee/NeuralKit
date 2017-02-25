//
//  FullyConnectedLayerTests.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.02.17.
//
//

import XCTest
@testable import NeuralKit

fileprivate let TestSetBase = "/Users/Palle/Developer/NeuralKit/Tests/NeuralKitTests/TestSets/"

class FullyConnectedLayerTests: XCTestCase
{
	var network: NeuralNetwork!
	
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
		
		let inputSamples = inputSampleRange
			.map {lines[$0]}
			.map {$0.components(separatedBy: ",")}
			.map {$0.flatMap{Float($0)}}
			.filter {$0.count > 0}
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
	
	func performRegressionTest(network net: NeuralNetwork, trainingSamples: [TrainingSample], testSamples: [InputSample], expected: [Matrix3])
	{
		var network = net
		for _ in 0 ..< 30_000
		{
			for sample in trainingSamples
			{
				network.train(sample, learningRate: 0.0004)
			}
		}
		
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
	
	
	func performBinaryClassificationTest(network net: NeuralNetwork, trainingSamples: [TrainingSample], testSamples: [InputSample], expected: [Matrix3])
	{
		var network = net
		for _ in 0 ..< 10_000
		{
			for sample in trainingSamples
			{
				network.train(sample, learningRate: 0.001)
			}
		}
		
		for (index, sample) in testSamples.enumerated()
		{
			let result = network.feedForward(sample.values)
			XCTAssertGreaterThan(result[0,0,0] * expected[index][0,0,0], 0) // If the sign matches, the classification is correct
		}
	}

	func testLinearRegression()
	{
		let network = NeuralNetwork(layers: [
			FullyConnectedLayer(weights: RandomWeightMatrix(width: 2, height: 1), activationFunction: tanh, activationDerivative: tanh_deriv)
		])
		for x in ["a", "b", "c", "d"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = regressionSamples(from: TestSetBase + "Regression/Linear/\(x)")
			performRegressionTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
	func testNonlinearRegression()
	{
		let	network = NeuralNetwork(layers: [
			FullyConnectedLayer(weights: RandomWeightMatrix(width: 2, height: 6), activationFunction: tanh, activationDerivative: tanh_deriv),
			FullyConnectedLayer(weights: RandomWeightMatrix(width: 7, height: 1), activationFunction: tanh, activationDerivative: tanh_deriv)
		])
		
		for x in ["a", "b", "c"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = regressionSamples(from: TestSetBase + "Regression/Nonlinear/\(x)")
			performRegressionTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
	func testLinearClassification()
	{
		let network = NeuralNetwork(
			layers: [FullyConnectedLayer(weights: RandomWeightMatrix(width: 3, height: 1), activationFunction: tanh, activationDerivative: tanh_deriv)],
			outputActivation: tanh,
			outputActivationDerivative: tanh_deriv
		)
		for x in ["a", "b", "c"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = binaryClassificationSamples(from: TestSetBase + "Classification/Linear/\(x)")
			performBinaryClassificationTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
	func testNonlinearClassification()
	{
		let network = NeuralNetwork(
			layers: [FullyConnectedLayer(weights: RandomWeightMatrix(width: 3, height: 6), activationFunction: tanh, activationDerivative: tanh_deriv),
			         FullyConnectedLayer(weights: RandomWeightMatrix(width: 7, height: 1), activationFunction: tanh, activationDerivative: tanh_deriv)],
			outputActivation: tanh,
			outputActivationDerivative: tanh_deriv
		)
		for x in ["a", "b", "c"]
		{
			let (trainingSamples, inputSamples, expectedOutputs) = binaryClassificationSamples(from: TestSetBase + "Classification/Nonlinear/\(x)")
			performBinaryClassificationTest(network: network, trainingSamples: trainingSamples, testSamples: inputSamples, expected: expectedOutputs)
		}
	}
	
}
