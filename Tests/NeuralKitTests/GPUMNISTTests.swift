//
//  GPUMNISTTests.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 15.04.17.
//
//

import NeuralKit
import XCTest

class GPUMNISTTests: XCTestCase
{
	@available(OSX 10.12, *)
	func testMNISTFullyConnected()
	{
		let (trainingSamples, testSamples) = MNISTTest.images(from: "/Users/Palle/Developer/MNIST/")
		
		let network = FeedForwardNeuralNetwork(
			layers: [
				ReshapingLayer(inputSize: (width: 28, height: 28, depth: 1), outputSize: (width: 1, height: 1, depth: 28*28)),
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 28*28+1, height: 500)),
				NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 500), activation: .tanh),
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 501, height: 250)),
				NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 250), activation: .tanh),
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 251, height: 10))
			],
			outputActivation: .softmax
		)!
		
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
			outputLayer: GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 10))
		)!
		
		let sema = DispatchSemaphore(value: 0)
		
		let trainer = GPUNetworkTrainingSession(
			network: gpuNetwork,
			batchSize: 1,
			optimizer: SGDOptimizer(learningRate: 0.01),
//			optimizer: MomentumOptimizer(learningRate: 0.01, momentum: 0.8),
//			optimizer: AdaGradOptimizer(learningRate: 0.01),
//			optimizer: AdaDeltaOptimizer(decay: 0.95),
			normalizers: [L2Normalizer(decay: 0.001)],
			sampleProvider: ArrayTrainingSampleProvider(samples: trainingSamples)
		)
		
		trainer.onFinishTraining = {sema.signal()}
		trainer.onBatchFinish = {
			_, epoch in
			if epoch % 1000 == 0
			{
				print("Epoch \(epoch)")
			}
		}
		trainer.train(epochs: 100_000)
		
		sema.wait()
		
		var correctCount = 0
		var wrongCount = 0
		
		for sample in testSamples
		{
			let input = GPUMatrix3(matrix: sample.values)
			let gpuResult = gpuNetwork.feedForward(input)
			
			let expectedIndex = argmax(sample.expected.values).1
			let actualIndex = argmax(gpuResult.values).1
			
			correctCount += expectedIndex == actualIndex ? 1 : 0
			wrongCount += expectedIndex == actualIndex ? 0 : 1
		}
		
		print("\(correctCount) correct, \(wrongCount) wrong, \(Float(correctCount) / Float(wrongCount + correctCount) * 100)% accuracy")
		
		// Assert 90% or higher accuracy
		XCTAssertGreaterThanOrEqual(Float(correctCount) / Float(wrongCount + correctCount), 0.9)
	}
	
	
	@available(OSX 10.12, *)
	func testMNISTConvNet()
	{
		let (trainingSamples, testSamples) = MNISTTest.images(from: "/Users/Palle/Developer/MNIST/")
		
		let conv1 = ConvolutionLayer(
			inputSize: (width: 28, height: 28, depth: 1),
			kernels: (0..<8).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 1, range: -0.04 ... 0.04)},
			bias: (0..<8).map{_ in Float(drand48()) * 0.08 - 0.04},
			horizontalStride: 1,
			verticalStride: 1,
			horizontalInset: 0,
			verticalInset: 0
		)
		let nonlinear1 = NonlinearityLayer(inputSize: (width: 24, height: 24, depth: 8), activation: .relu)
		
		let pool1 = PoolingLayer(inputSize: (width: 24, height: 24, depth: 8), outputSize: (width: 12, height: 12, depth: 8))
		
		let conv2 = ConvolutionLayer(
			inputSize: (width: 12, height: 12, depth: 8),
			kernels: (0..<16).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 8, range: -0.005 ... 0.005)},
			bias: (0..<16).map{_ in Float(drand48()) * 0.01 - 0.005},
			horizontalStride: 1,
			verticalStride: 1,
			horizontalInset: -2,
			verticalInset: -2
		)
		let nonlinear2 = NonlinearityLayer(inputSize: (width: 12, height: 12, depth: 16), activation: .relu)
		
		let pool2 = PoolingLayer(inputSize: (width: 12, height: 12, depth: 16), outputSize: (width: 4, height: 4, depth: 16))
		
		let reshape = ReshapingLayer(inputSize: (width: 4, height: 4, depth: 16), outputSize: (width: 1, height: 1, depth: 256))
		
		let fullyConnected = FullyConnectedLayer(weights: RandomWeightMatrix(width: 257, height: 10, range: -0.004 ... 0.004))
		
		let network = FeedForwardNeuralNetwork(layers: [conv1, nonlinear1, pool1, conv2, nonlinear2, pool2, reshape, fullyConnected], outputActivation: .softmax)!
		
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
			outputLayer: GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 10))
		)!
		
		//		let epochs = 10_000_000
		//
		//		var time = CACurrentMediaTime()
		
		let numberFormatter = NumberFormatter()
		numberFormatter.maximumSignificantDigits = 5
		numberFormatter.maximumFractionDigits = 3
		numberFormatter.maximumIntegerDigits = 10
		numberFormatter.localizesFormat = false
		
		let session = GPUNetworkTrainingSession(
			network: gpuNetwork,
			batchSize: 20,
			optimizer: MomentumOptimizer(learningRate: 0.01, momentum: 0.0),
//			optimizer: AdaGradOptimizer(learningRate: 0.01),
//			optimizer: AdaDeltaOptimizer(decay: 0.95),
			normalizers: [L2Normalizer(decay: 0.001)],
			sampleProvider: CachedArrayTrainingSampleProvider(samples: trainingSamples)
		)
		
		let sema = DispatchSemaphore(value: 0)
		
		session.onFinishTraining = {sema.signal()}
		session.onBatchFinish = {
			_, epoch in
			if epoch % 100 == 0
			{
				print("Epoch \(epoch)")
			}
		}
		session.train(epochs: 10_000)
		sema.wait()
		
		for (index, layer) in gpuNetwork.layers.enumerated()
		{
			if let gradient = layer.gradient?.asMatrix()
			{
				let gradientImages = CGImage.make(from: gradient)
				gradientImages.enumerated().forEach
				{ i, image in
					image.write(to: URL(fileURLWithPath: "/Users/Palle/Desktop/gradient_images/grad_\(index)_\(i).png"))
				}
			}
			
			if let activation = layer.activation?.asMatrix()
			{
				let activationImages = CGImage.make(from: activation)
				activationImages.enumerated().forEach
				{ i, image in
					image.write(to: URL(fileURLWithPath: "/Users/Palle/Desktop/gradient_images/act_\(index)_\(i).png"))
				}
			}
		}
		
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
	
	@available(OSX 10.12, *)
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
			outputLayer: GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 10))
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

}
