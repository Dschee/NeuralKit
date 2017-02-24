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
	
	func samples(from file: String) -> ([TrainingSample], [InputSample])
	{
		guard let content = try? String(contentsOf: URL(fileURLWithPath: file)) else { return ([],[]) }
		let lines = content.components(separatedBy: "\n")
		let trainingSampleRange:CountableRange<Int> = 0 ..< (lines.index(where: {$0.hasPrefix("0,0")}) ?? 1)
		let trainingSamples = trainingSampleRange.map
		{ line -> TrainingSample in
			let components = lines[line].components(separatedBy: ",").flatMap{Float($0)}
			let inputs = Array<Float>(components.dropLast())
			let output = components.last!
			let inputMatrix = Matrix3(values: inputs, width: 1, height: 1, depth: inputs.count)
			let outputMatrix = Matrix3(values: [output], width: 1, height: 1, depth: 1)
			return TrainingSample(input: inputMatrix, expected: outputMatrix)
		}
		let inputSampleRange = ((lines.index(where: {$0.hasPrefix("0,0")}) ?? 1) + 1) ..< lines.count
		let inputSamples = inputSampleRange.map
		{ line -> InputSample in
			let components = lines[line].components(separatedBy: ",").flatMap{Float($0)}
			let inputMatrix = Matrix3(values: components, width: 1, height: 1, depth: components.count)
			return InputSample(values: inputMatrix)
		}
		
		return (trainingSamples, inputSamples)
	}
	
    override func setUp()
	{
        super.setUp()
    }
    
    override func tearDown()
	{
        super.tearDown()
    }

	func testLinearRegression()
	{
		var network = NeuralNetwork(layers: [
			FullyConnectedLayer(weights: RandomWeightMatrix(width: 2, height: 1), activationFunction: tanh, activationDerivative: tanh_deriv)
		])
		let (trainingSamples, inputSamples) = samples(from: TestSetBase + "Regression/Linear/a.input.txt")
		print(trainingSamples, inputSamples)
		for _ in 0 ..< 10_000
		{
			for sample in trainingSamples
			{
				network.train(sample, learningRate: 0.001)
			}
		}
		
		for sample in inputSamples
		{
			print(network.feedForward(sample.values)[0,0,0])
		}
	}
	
}
