//
//  ReverseMNISTTest.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 01.03.17.
//
//

import Foundation
import XCTest
@testable import NeuralKit

class ReverseMNISTTest: XCTestCase
{

	static func readSamples(from bytes: [UInt8], labels: [UInt8], count: Int) -> [TrainingSample]
	{
		let imageOffset = 16
		let labelOffset = 8
		
		let imageWidth = 28
		let imageHeight = 28
		
		var samples:[TrainingSample] = []
		
		for i in 0 ..< count
		{
			let offset = imageOffset + imageWidth * imageHeight * i
			let pixelData = bytes[offset ..< (offset + imageWidth * imageHeight)]
				.map{Float($0)/128-1}
			
			let sampleMatrix = Matrix3(values: pixelData, width: imageWidth, height: imageHeight, depth: 1)
			
			let label = Int(labels[labelOffset + i])
			
			let input = TrainingSample.encodeOneHot(count: 10, target: label, baseValue: 0, hotValue: 1)
			
			let sample = TrainingSample(values: input, expected: sampleMatrix)
			samples.append(sample)
		}
		
		return samples
	}
	
	static func images(from path: String) -> ([TrainingSample], [TrainingSample])
	{
		guard
			let trainingData = try? Data(contentsOf: URL(fileURLWithPath: path + "train-images-idx3-ubyte")),
			let trainingLabelData = try? Data(contentsOf: URL(fileURLWithPath: path + "train-labels-idx1-ubyte")),
			let testingData = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-images-idx3-ubyte")),
			let testingLabelData = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-labels-idx1-ubyte"))
			else
		{
			return ([],[])
		}
		
		let trainingBytes = Array<UInt8>(trainingData)
		let trainingLabels = Array<UInt8>(trainingLabelData)
		let testingBytes = Array<UInt8>(testingData)
		let testingLabels = Array<UInt8>(testingLabelData)
		
		let trainingSampleCount = 60_000
		let testingSampleCount = 10_000
		
		let trainingSamples = readSamples(from: trainingBytes, labels: trainingLabels, count: trainingSampleCount)
		let testingSamples = readSamples(from: testingBytes, labels: testingLabels, count: testingSampleCount)
		
		return (trainingSamples,testingSamples)
	}
	
	func testMNISTGeneration()
	{
		let (trainingSamples, testSamples) = ReverseMNISTTest.images(from: "/Users/Palle/Developer/MNIST/")
		
		let inputLayer = FullyConnectedLayer(weights: RandomWeightMatrix(width: 11, height: 200), activationFunction: .linear)
		let hiddenLayer1 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 201, height: 500), activationFunction: .tanh)
		let hiddenLayer2 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 501, height: 800), activationFunction: .tanh)
		let hiddenLayer3 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 801, height: 784), activationFunction: .tanh)
		
		var network = NeuralNetwork(layers: [inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3], outputActivation: .tanh)!
		
		let epochs = 300_000
		
		for epoch in 0 ..< epochs
		{
			let sample = trainingSamples[Int(arc4random_uniform(UInt32(trainingSamples.count)))]
			let error = network.train(sample, learningRate: 0.005 * pow(0.999996, Float(epoch))/*, annealingRate: epoch % 300 == 0 ? 0.002 * pow(0.99999, Float(epoch)) : 0*/)
			
			if epoch % 1000 == 0
			{
				print("epoch \(epoch) of \(epochs): \(error * 100)% error")
			}
		}
		
		var correctCount = 0
		var wrongCount = 0
		
		for sample in testSamples
		{
			let result = network.feedForward(sample.values)
			
			let expectedIndex = maxi(sample.expected.values).1
			let actualIndex = maxi(result.values).1
			
			XCTAssertEqual(expectedIndex, actualIndex)
			correctCount += expectedIndex == actualIndex ? 1 : 0
			wrongCount += expectedIndex == actualIndex ? 0 : 1
		}
		
		print("\(correctCount) correct, \(wrongCount) wrong, \(Float(correctCount) / Float(wrongCount + correctCount) * 100)% accuracy")
		
//		let trainedLayerMatrices = network.layers
//			.flatMap{$0 as? FullyConnectedLayer}
//			.map{$0.weights}
//			.map{$0.serialize()}
//			.map{$0.object}
//		
//		let json:JSON =
//		[
//				"layers": trainedLayerMatrices
//		]
//		
//		guard let rawString = json.rawString() else
//		{
//			return
//		}
//		
//		do
//		{
//			try rawString.write(toFile: "/Users/Palle/Desktop/mnist_reverse.network", atomically: true, encoding: .ascii)
//		}
//		catch
//		{
//			fatalError(String(describing: error))
//		}
	}

}
