//
//  ConvolutionalNetworkTest.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 05.04.17.
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
@testable import NeuralKitGPU

class NetworkPerformanceTests: XCTestCase
{
	func testCPUConvNetPerformance()
	{
		let network = FeedForwardNeuralNetwork(
			layers: [
				ConvolutionLayer(
					inputSize: (width: 128, height: 128, depth: 3),
					kernels: (0..<8).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 3)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 8).values,
					horizontalStride: 1,
					verticalStride: 1,
					horizontalInset: -2,
					verticalInset: -2
				),
				NonlinearityLayer(inputSize: (width: 128, height: 128, depth: 8), activation: .relu),
				PoolingLayer(inputSize: (width: 128, height: 128, depth: 8), outputSize: (width: 64, height: 64, depth: 8)),
				
				ConvolutionLayer(
					inputSize: (width: 64, height: 64, depth: 8),
					kernels: (0..<16).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 8)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 16).values,
					horizontalStride: 1,
					verticalStride: 1,
					horizontalInset: -2,
					verticalInset: -2
				),
				NonlinearityLayer(inputSize: (width: 64, height: 64, depth: 16), activation: .relu),
				PoolingLayer(inputSize: (width: 64, height: 64, depth: 16), outputSize: (width: 32, height: 32, depth: 16)),
				
				ConvolutionLayer(
					inputSize: (width: 32, height: 32, depth: 16),
					kernels: (0..<25).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 16)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 25).values,
					horizontalStride: 1,
					verticalStride: 1,
					horizontalInset: -2,
					verticalInset: -2
				),
				NonlinearityLayer(inputSize: (width: 32, height: 32, depth: 25), activation: .relu),
				PoolingLayer(inputSize: (width: 32, height: 32, depth: 25), outputSize: (width: 16, height: 16, depth: 25)),
				
				ConvolutionLayer(
					inputSize: (width: 16, height: 16, depth: 25),
					kernels: (0..<50).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 25)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 50).values,
					horizontalStride: 1,
					verticalStride: 1,
					horizontalInset: -2,
					verticalInset: -2
				),
				NonlinearityLayer(inputSize: (width: 16, height: 16, depth: 50), activation: .relu),
				PoolingLayer(inputSize: (width: 16, height: 16, depth: 50), outputSize: (width: 8, height: 8, depth: 50)),
				
				ConvolutionLayer(
					inputSize: (width: 8, height: 8, depth: 50),
					kernels: (0..<100).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 25)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 100).values,
					horizontalStride: 1,
					verticalStride: 1,
					horizontalInset: 0,
					verticalInset: 0
				),
				NonlinearityLayer(inputSize: (width: 4, height: 4, depth: 100), activation: .relu),
				
				ReshapingLayer(inputSize: (width: 4, height: 4, depth: 100), outputSize: (width: 1, height: 1, depth: 1600)),
				
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 1601, height: 800)),
				
				NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 800), activation: .tanh),
				
				FullyConnectedLayer(weights: RandomWeightMatrix(width: 801, height: 100))
			],
			outputLayer: NonlinearityLayer(inputSize: (width: 1, height: 1, depth: 100), activation: .softmax)
		)!
		
		let samples = (0..<100).map{_ in RandomWeightMatrix(width: 128, height: 128, depth: 3)}
		
		self.measure
		{
			for sample in samples
			{
				_ = network.feedForward(sample)
			}
			
			print("measure completed")
		}
	}
	
	@available(OSX 10.12, *)
	func testGPUConvNetworkPerformance()
	{
		let network = GPUFeedForwardNeuralNetwork(
			layers: [
				GPUConvolutionLayer(
					inputSize: (width: 128, height: 128, depth: 3),
					kernels: (0..<8).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 3)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 8).values,
					horizontalInset: -2,
					verticalInset: -2
				),
				GPUNonlinearityLayer(inputSize: (width: 128, height: 128, depth: 8), activation: .relu),
				GPUPoolingLayer(inputSize: (width: 128, height: 128, depth: 8), outputSize: (width: 64, height: 64, depth: 8)),
				
				GPUConvolutionLayer(
					inputSize: (width: 64, height: 64, depth: 8),
					kernels: (0..<16).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 8)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 16).values,
					horizontalInset: -2,
					verticalInset: -2
				),
				GPUNonlinearityLayer(inputSize: (width: 64, height: 64, depth: 16), activation: .relu),
				GPUPoolingLayer(inputSize: (width: 64, height: 64, depth: 16), outputSize: (width: 32, height: 32, depth: 16)),
				
				GPUConvolutionLayer(
					inputSize: (width: 32, height: 32, depth: 16),
					kernels: (0..<25).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 16)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 25).values,
					horizontalInset: -2,
					verticalInset: -2
				),
				GPUNonlinearityLayer(inputSize: (width: 32, height: 32, depth: 25), activation: .relu),
				GPUPoolingLayer(inputSize: (width: 32, height: 32, depth: 25), outputSize: (width: 16, height: 16, depth: 25)),
				
				GPUConvolutionLayer(
					inputSize: (width: 16, height: 16, depth: 25),
					kernels: (0..<50).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 25)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 50).values,
					horizontalInset: -2,
					verticalInset: -2
				),
				GPUNonlinearityLayer(inputSize: (width: 16, height: 16, depth: 50), activation: .relu),
				GPUPoolingLayer(inputSize: (width: 16, height: 16, depth: 50), outputSize: (width: 8, height: 8, depth: 50)),
				
				GPUConvolutionLayer(
					inputSize: (width: 8, height: 8, depth: 50),
					kernels: (0..<100).map{_ in RandomWeightMatrix(width: 5, height: 5, depth: 25)},
					bias: RandomWeightMatrix(width: 1, height: 1, depth: 100).values,
					horizontalInset: 0,
					verticalInset: 0
				),
				GPUNonlinearityLayer(inputSize: (width: 4, height: 4, depth: 100), activation: .relu),
				
				GPUReshapingLayer(inputSize: (width: 4, height: 4, depth: 100), outputSize: (width: 1, height: 1, depth: 1600)),
				
				GPUFullyConnectedLayer(weights: RandomWeightMatrix(width: 1601, height: 800)),
				
				GPUNonlinearityLayer(inputSize: (width: 1, height: 1, depth: 800), activation: .tanh),
				
				GPUFullyConnectedLayer(weights: RandomWeightMatrix(width: 801, height: 100))
			],
			outputLayer: GPUSoftmaxLayer(inputSize: (width: 1, height: 1, depth: 100))
		)!
		
		let samples = (0..<1000).map{_ in RandomWeightMatrix(width: 128, height: 128, depth: 3)}.map{GPUMatrix3(matrix: $0)}
		
		self.measure
		{
			for sample in samples
			{
				_ = network.feedForward(sample)
			}
		}
	}
}
