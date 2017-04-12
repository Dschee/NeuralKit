//
//  MNISTTest.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 25.02.17.
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
import XCTest
import Cocoa
@testable import NeuralKit

class MNISTTest: XCTestCase
{
	lazy var library: MTLLibrary =
	{
#if DEBUG
		return try! GPUGlobalDevice.makeLibrary(filepath: "/Users/Palle/Library/Developer/Xcode/DerivedData/NeuralKit-gqvgbrxdpkclopbfiglfaeqqojot/Build/Products/Debug/NeuralKit.framework/Versions/A/Resources/default.metallib")
#else
		return try! GPUGlobalDevice.makeLibrary(filepath: "/Users/Palle/Library/Developer/Xcode/DerivedData/NeuralKit-gqvgbrxdpkclopbfiglfaeqqojot/Build/Products/Release/NeuralKit.framework/Versions/A/Resources/default.metallib")
#endif
	}()
	
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
				.map{Float($0)/256}
			
			let label = Int(labels[labelOffset + i])
			
			let sampleMatrix = Matrix3(values: pixelData, width: imageWidth, height: imageHeight, depth: 1)
			let sample = TrainingSample(values: sampleMatrix, outputCount: 10, targetIndex: label, baseValue: 0.0, hotValue: 1.0)
			samples.append(sample)
			
//			for _ in 0 ..< 8
//			{
//				let offsetX = Int(arc4random_uniform(16))-8
//				let offsetY = Int(arc4random_uniform(8))-4
//				let randomizedSampleMatrix = sampleMatrix[x: offsetX, y: offsetY, z: 0, width: 28, height: 28, depth: 1]
//				
//				let label = Int(labels[labelOffset + i])
//				
//				let sample = TrainingSample(values: randomizedSampleMatrix, outputCount: 10, targetIndex: label, baseValue: -1.0, hotValue: 1.0)
//				samples.append(sample)
//			}
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
	
	func testMNISTClassification()
	{
		let (trainingSamples, testSamples) = MNISTTest.images(from: "/Users/Palle/Developer/MNIST/")
		
		let reshapingLayer = ReshapingLayer(inputSize: (width: 28, height: 28, depth: 1), outputSize: (width: 1, height: 1, depth: 28*28))
		let inputLayer = FullyConnectedLayer(weights: RandomWeightMatrix(width: 28*28+1, height: 800))
		let tanh1 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 800), activation: .tanh)
		let hiddenLayer2 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 801, height: 500))
		let tanh2 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 500), activation: .tanh)
		let hiddenLayer3 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 501, height: 200))
		let tanh3 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 200), activation: .tanh)
		let hiddenLayer4 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 201, height: 10))
		
		var network = FeedForwardNeuralNetwork(layers: [reshapingLayer, inputLayer, tanh1, hiddenLayer2, tanh2, hiddenLayer3, tanh3, hiddenLayer4], outputActivation: .softmax)!
		
		let epochs = 200_000
		
		var time = CACurrentMediaTime()
		
		var gpuNetwork = GPUFeedForwardNeuralNetwork(
			layers: network.layers.flatMap
				{ l in
					if let layer = l as? ReshapingLayer
					{
						return GPUReshapingLayer(inputSize: layer.inputSize, outputSize: layer.outputSize)
					}
					else if let layer = l as? FullyConnectedLayer
					{
						return GPUFullyConnectedLayer(weights: layer.weights, updateMethod: .sgd)
					}
					else if let layer = l as? NonlinearityLayer
					{
						return GPUNonlinearityLayer(inputSize: layer.inputSize, activation: layer.activation)
					}
					else
					{
						return nil
					}
			},
			outputLayer: GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 10)),//GPUNonlinearityLayer(inputSize: (width: 1, height: 1, depth: 10), activation: .tanh), //GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 10)),
			library: library
		)!
		
//		for epoch in 0 ..< epochs
//		{
//			let sample = trainingSamples[Int(arc4random_uniform(UInt32(trainingSamples.count)))]
////			let error = network.train(sample, learningRate: 0.005 * pow(0.999996, Float(epoch))/*, annealingRate: epoch % 300 == 0 ? 0.002 * pow(0.99999, Float(epoch)) : 0*/)
//			let input = GPUMatrix3(matrix: sample.values, isShared: true)
//			let output = GPUMatrix3(matrix: sample.expected, isShared: true)
//			
////			let error = gpuNetwork.train((input: input, expected: output),	learningRate: 0.005 * pow(0.999996, Float(epoch)), deltaDecay: 0.98)
////			let error = Float.nan
//			let cpuError = network.train(sample,							learningRate: 0.005 * pow(0.999996, Float(epoch)))
//			
////			print(error)
//			
//			if epoch % 1000 == 0
//			{
//				let newTime = CACurrentMediaTime()
////				print("epoch \(epoch) of \(epochs): \(error * 100)% error, \(cpuError * 100) error on CPU, duration: \(newTime - time) seconds.")
//				time = newTime
//			}
//		}
		
		let time1 = CACurrentMediaTime()
		
		let gpuSamples = testSamples.map{$0.values}.map{GPUMatrix3(matrix: $0)}
		
		let time2 = CACurrentMediaTime()
		
		print("Copy time: \(time2 - time1) seconds")
		
		var correctCount = 0
		var wrongCount = 0
		
		for (sample, input) in zip(testSamples, gpuSamples)
		{
//			let result = network.feedForward(sample.values)
			let gpuResult = gpuNetwork.feedForward(input)
			
			let expectedIndex = argmax(sample.expected.values).1
			let actualIndex = argmax(gpuResult.values).1
			
//			XCTAssertEqual(expectedIndex, actualIndex)
			correctCount += expectedIndex == actualIndex ? 1 : 0
			wrongCount += expectedIndex == actualIndex ? 0 : 1
			
//			for (cpuValue, gpuValue) in zip(gpuResult.values, gpuResult.values)
//			{
//				XCTAssertEqualWithAccuracy(cpuValue, gpuValue, accuracy: 0.001)
//			}
		}
		
		print("\(correctCount) correct, \(wrongCount) wrong, \(Float(correctCount) / Float(wrongCount + correctCount) * 100)% accuracy")
		
		// Assert 90% or higher accuracy
		XCTAssertGreaterThanOrEqual(Float(correctCount) / Float(wrongCount + correctCount), 0.9)
		
		var gpuCorrectCount = 0
		var gpuWrongCount = 0
		
		for (sample, input) in zip(testSamples, gpuSamples)
		{
			let result = gpuNetwork.feedForward(input)
//			print(result.values)
			let expectedIndex = argmax(sample.expected.values).1
			let actualIndex = argmax(result.values).1
			
//			XCTAssertEqual(expectedIndex, actualIndex)
			gpuCorrectCount += expectedIndex == actualIndex ? 1 : 0
			gpuWrongCount += expectedIndex == actualIndex ? 0 : 1
		}
		
		let time3 = CACurrentMediaTime()
		
		print("Eval time: \(time3 - time2) seconds")
		
		XCTAssertEqual(gpuCorrectCount, correctCount)
		XCTAssertEqual(gpuWrongCount, wrongCount)
	}
	
	func testMNISTMetalFeedForwardPerformance()
	{
		let (_, testSamples) = MNISTTest.images(from: "/Users/Palle/Developer/MNIST/")
		
		let reshapingLayer = ReshapingLayer(inputSize: (width: 28, height: 28, depth: 1), outputSize: (width: 1, height: 1, depth: 28*28))
		let inputLayer = FullyConnectedLayer(weights: RandomWeightMatrix(width: 28*28+1, height: 800))
		let tanh1 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 800), activation: .tanh)
		let hiddenLayer2 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 801, height: 500))
		let tanh2 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 500), activation: .tanh)
		let hiddenLayer3 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 501, height: 200))
		let tanh3 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 200), activation: .tanh)
		let hiddenLayer4 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 201, height: 10))
		
		let network = FeedForwardNeuralNetwork(layers: [reshapingLayer, inputLayer, tanh1, hiddenLayer2, tanh2, hiddenLayer3, tanh3, hiddenLayer4], outputActivation: .softmax)!
		
		let gpuNetwork = GPUFeedForwardNeuralNetwork(
			layers: network.layers.flatMap
				{ l in
					if let layer = l as? ReshapingLayer
					{
						return GPUReshapingLayer(inputSize: layer.inputSize, outputSize: layer.outputSize)
					}
					else if let layer = l as? FullyConnectedLayer
					{
						return GPUFullyConnectedLayer(weights: layer.weights)
					}
					else if let layer = l as? NonlinearityLayer
					{
						return GPUNonlinearityLayer(inputSize: layer.inputSize, activation: layer.activation)
					}
					else
					{
						return nil
					}
			},
			outputLayer: GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 10)),
			library: library
		)!
		
		let time1 = CACurrentMediaTime()
		
		let gpuSamples = testSamples.map{$0.values}.map{GPUMatrix3(matrix: $0)}
		
		let time2 = CACurrentMediaTime()
		
		print("Copy time: \(time2 - time1) seconds")
		
		var gpuCorrectCount = 0
		var gpuWrongCount = 0
		
		for (sample, input) in zip(testSamples, gpuSamples)
		{
			let result = gpuNetwork.feedForward(input)
			//			print(result.values)
			let expectedIndex = argmax(sample.expected.values).1
			let actualIndex = argmax(result.values).1
			
			//			XCTAssertEqual(expectedIndex, actualIndex)
			gpuCorrectCount += expectedIndex == actualIndex ? 1 : 0
			gpuWrongCount += expectedIndex == actualIndex ? 0 : 1
		}
		
		let time3 = CACurrentMediaTime()
		
		print("Eval time: \(time3 - time2) seconds")
	}
	
	func testMNISTConvNet()
	{
		let (trainingSamples, testSamples) = MNISTTest.images(from: "/Users/Palle/Developer/MNIST/")
		
		let conv1 = ConvolutionLayer(
			inputSize: (width: 28, height: 28, depth: 1),
			kernels: (0..<8).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 1, range: -0.001 ... 0.001)},
			bias: (0..<8).map{_ in Float(drand48()) * 0.002 - 0.001},
			horizontalStride: 1,
			verticalStride: 1,
			horizontalInset: 0,
			verticalInset: 0
		)
		let nonlinear1 = NonlinearityLayer(inputSize: (width: 24, height: 24, depth: 8), activation: .relu)
		
		let pool1 = PoolingLayer(inputSize: (width: 24, height: 24, depth: 8), outputSize: (width: 12, height: 12, depth: 8))
		
		let conv2 = ConvolutionLayer(
			inputSize: (width: 12, height: 12, depth: 8),
			kernels: (0..<16).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 8, range: -0.001 ... 0.001)},
			bias: (0..<16).map{_ in Float(drand48()) * 0.002 - 0.001},
			horizontalStride: 1,
			verticalStride: 1,
			horizontalInset: -2,
			verticalInset: -2
		)
		let nonlinear2 = NonlinearityLayer(inputSize: (width: 12, height: 12, depth: 16), activation: .relu)
		
		let pool2 = PoolingLayer(inputSize: (width: 12, height: 12, depth: 16), outputSize: (width: 4, height: 4, depth: 16))
		
		let reshape = ReshapingLayer(inputSize: (width: 4, height: 4, depth: 16), outputSize: (width: 1, height: 1, depth: 256))
		
		let fullyConnected = FullyConnectedLayer(weights: RandomWeightMatrix(width: 257, height: 10))
		
		let network = FeedForwardNeuralNetwork(layers: [conv1, nonlinear1, pool1, conv2, nonlinear2, pool2, reshape, fullyConnected], outputActivation: .softmax)!
		
		var gpuNetwork = GPUFeedForwardNeuralNetwork(
			layers: network.layers.flatMap
			{ l in
				if let layer = l as? ReshapingLayer
				{
					return GPUReshapingLayer(inputSize: layer.inputSize, outputSize: layer.outputSize)
				}
				else if let layer = l as? FullyConnectedLayer
				{
					return GPUFullyConnectedLayer(weights: layer.weights, updateMethod: .momentum)
				}
				else if let layer = l as? NonlinearityLayer
				{
					return GPUNonlinearityLayer(inputSize: layer.inputSize, activation: layer.activation)
				}
				else if let layer = l as? ConvolutionLayer
				{
					return GPUConvolutionLayer(inputSize: layer.inputSize, kernels: layer.kernels, bias: layer.bias, horizontalInset: layer.horizontalInset, verticalInset: layer.verticalInset)
				}
				else if let layer = l as? PoolingLayer
				{
					return GPUPoolingLayer(inputSize: layer.inputSize, outputSize: layer.outputSize)
				}
				else
				{
					return nil
				}
			},
			outputLayer: GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 10)),
			library: library
		)!
		
		let epochs = 10_000_000
		
		var time = CACurrentMediaTime()
		
		let numberFormatter = NumberFormatter()
		numberFormatter.maximumSignificantDigits = 5
		numberFormatter.maximumFractionDigits = 3
		numberFormatter.maximumIntegerDigits = 10
		numberFormatter.localizesFormat = false
		
//		print("Copying inputs...")
		
//		let gpuInputs = trainingSamples.map{$0.values}.map{GPUMatrix3(matrix: $0, isShared: false)}
		
//		print("Copying outputs...")
//		
//		let gpuOutputs = trainingSamples.map{$0.expected}.map{GPUMatrix3(matrix: $0, isShared: true)}
//		
//		print("Done.")
		
//		for epoch in 0 ..< epochs
//		{
//			let index = Int(arc4random_uniform(UInt32(trainingSamples.count)))
//			let sample = trainingSamples[index]
////			let input = gpuInputs[index]
//			let input = GPUMatrix3(matrix: sample.values, isShared: true)
//			let output = GPUMatrix3(matrix: sample.expected, isShared: true)
////			let output = gpuOutputs[index]
////			let error = network.train(sample, learningRate: 0.01, annealingRate: 0, momentum: 0, decay: 0)
////			let error = gpuNetwork.train((input: input, expected: output), learningRate: 0.01, momentum: 0.9, decay: 0.001)
//			
//			if epoch % 1000 == 0
//			{
//				let newTime = CACurrentMediaTime()
//				print("epoch \(epoch) of \(epochs): \(numberFormatter.string(from: (error * 100) as NSNumber) ?? "")% error, duration: \(numberFormatter.string(from: (newTime - time) as NSNumber) ?? "") seconds.")
//				time = newTime
//			}
//			
//			if error.isNaN
//			{
//				XCTFail("NaN error")
//				return
//			}
//		}
		
		var correctCount = 0
		var wrongCount = 0
		
		for sample in testSamples
		{
			let input = GPUMatrix3(matrix: sample.values)
			let result = gpuNetwork.feedForward(input)
			
			let expectedIndex = argmax(sample.expected.values).1
			let actualIndex = argmax(result.values).1
			
			correctCount += expectedIndex == actualIndex ? 1 : 0
			wrongCount += expectedIndex == actualIndex ? 0 : 1
		}
		
		XCTAssertGreaterThanOrEqual(Float(correctCount) / Float(wrongCount + correctCount), 0.9)
		print("\(correctCount) correct, \(wrongCount) wrong, \(numberFormatter.string(from: (Double(correctCount) / Double(wrongCount + correctCount) * 100) as NSNumber) ?? "")% accuracy")
	}
}
