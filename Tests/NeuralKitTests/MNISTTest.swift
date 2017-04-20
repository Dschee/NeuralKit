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
import MatrixVector

class MNISTTest: XCTestCase
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
	
	
	@available(OSX 10.12, *)
	func testMNISTClassification()
	{
		let (trainingSamples, testSamples) = MNISTTest.images(from: "/Users/Palle/Developer/MNIST/")
		
		let reshapingLayer = ReshapingLayer(inputSize: (width: 28, height: 28, depth: 1), outputSize: (width: 1, height: 1, depth: 28*28))
		let inputLayer = FullyConnectedLayer(weights: RandomWeightMatrix(width: 28*28+1, height: 800))
		let tanh1 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 800), activation: .relu)
		let hiddenLayer2 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 801, height: 500))
		let tanh2 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 500), activation: .relu)
		let hiddenLayer3 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 501, height: 200))
		let tanh3 = NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 200), activation: .relu)
		let hiddenLayer4 = FullyConnectedLayer(weights: RandomWeightMatrix(width: 201, height: 10))
		
		var network = FeedForwardNeuralNetwork(
			layers: [
				reshapingLayer,
				inputLayer,
				tanh1,
				hiddenLayer2,
				tanh2,
				hiddenLayer3,
				tanh3,
				hiddenLayer4
			],
			outputLayer: NonlinearityLayer(
				inputSize: (
					width: 1,
					height: 1,
					depth: 10
				),
				activation: .softmax
			)
		)!
		
		let epochs = 100_000
		
		let sema = DispatchSemaphore(value: 0)
		
		let session = NetworkTrainingSession(
			network: network,
			batchSize: 1,
			optimizer: SGDOptimizer(learningRate: 0.005),
			normalizers: [],
			sampleProvider: ArrayTrainingSampleProvider(samples: trainingSamples)
		)
		
		session.onBatchFinish = {
			error, epoch in
			
			if epoch % 1000 == 0
			{
				print("Epoch \(epoch)")
			}
		}
		
		session.onFinishTraining = {
			sema.signal()
		}
		
		session.train(epochs: epochs)
		sema.wait()
		
		network = session.network
		
		var correctCount = 0
		var wrongCount = 0
		
		for sample in testSamples
		{
			if argmax(sample.expected.values).1 == argmax(network.feedForward(sample.values).values).1
			{
				correctCount += 1
			}
			else
			{
				wrongCount += 1
			}
		}
		
		print("\(correctCount) correct, \(wrongCount) wrong, \(Float(correctCount) / Float(wrongCount + correctCount) * 100)% accuracy")
		
		// Assert 90% or higher accuracy
		XCTAssertGreaterThanOrEqual(Float(correctCount) / Float(wrongCount + correctCount), 0.9)
	}
	
	
}
