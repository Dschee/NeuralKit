//
//  SevenSegmentDisplayTests.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 12.05.17.
//
//

import XCTest
@testable import NeuralKit
@testable import MatrixVector
@testable import NeuralKit
@testable import Serialization

@available(OSX 10.12, *)
class SevenSegmentDisplayTests: XCTestCase
{
	func test7SegmentDisplay()
	{
		let sampleData:[[Float]] = [
			[1,1,1,0,1,1,1],
			[0,0,1,0,0,1,0],
			[1,0,1,1,1,0,1],
			[1,0,1,1,0,1,1],
			[0,1,1,1,0,1,0],
			[1,1,0,1,0,1,1],
			[1,1,0,1,1,1,1],
			[1,0,1,0,0,1,0],
			[1,1,1,1,1,1,1],
			[1,1,1,1,0,1,1]
		]
		
		let trainingSamples = sampleData.enumerated().map { (offset, sample) -> TrainingSample in
			TrainingSample(
				values: Matrix3(
					values: sample,
					width: 1,
					height: 1,
					depth: 7
				),
				outputCount: 10,
				targetIndex: offset,
				baseValue: 0,
				hotValue: 1
			)
		}
		
		trainingSamples.forEach { sample in
			print(sample.expected.values.map(String.init).joined(separator: ", "))
		}
		
		var network = FeedForwardNeuralNetwork(
			layers: [
				FullyConnectedLayer(inputDepth: 7, outputDepth: 10),
				NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 10), activation: .tanh),
				FullyConnectedLayer(inputDepth: 10, outputDepth: 10)
			],
			outputLayer: NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 10), activation: .softmax)
		)!
		
		let session = NetworkTrainingSession(
			network: network,
			batchSize: 1,
			optimizer: SGDOptimizer(learningRate: 0.003),
			normalizers: [],
			sampleProvider: ArrayTrainingSampleProvider(samples: trainingSamples)
		)
		let sema = DispatchSemaphore(value: 0)
		
		session.onFinishTraining = {
			sema.signal()
		}
		session.train(epochs: 100_000)
		sema.wait()
		
		network = session.network
		
		let formatter = NumberFormatter()
		formatter.maximumIntegerDigits = 1
		formatter.minimumIntegerDigits = 1
		formatter.minimumFractionDigits = 4
		formatter.maximumFractionDigits = 4
		
		for (expected, sample) in trainingSamples.enumerated()
		{
			let output = network.feedForward(sample.values).values
			print(output.map { formatter.string(from: NSNumber(floatLiteral: Double($0)))! }.joined(separator: ", "))
			XCTAssertEqual(expected, argmax(output).1)
		}
	}
}
