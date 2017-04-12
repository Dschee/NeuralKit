//
//  GPUWeightOptimizer.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 11.04.17.
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
import Metal

public enum GPUTensor
{
	case vector(MTLBuffer, length: Int)
	case matrix(GPUMatrix)
	case matrix3(GPUMatrix3)
}

public protocol Optimizer
{
	associatedtype OptimizerData
	
	func update(weights: [GPUTensor], gradients: [GPUTensor], data: OptimizerData?) -> OptimizerData
}

public struct SGDOptimizer: Optimizer
{
	public typealias OptimizerData = ()
	
	public var learningRate: Float
	
	
	
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], data: ()?) -> ()
	{
		
	}
}

public struct MomentumOptimizer: Optimizer
{
	public typealias OptimizerData = [GPUTensor]
	
	public var learningRate: Float
	public var momentum: Float
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], data: [GPUTensor]?) -> [GPUTensor]
	{
		return []
	}
}

public struct AdaGradOptimizer: Optimizer
{
	public typealias OptimizerData = [GPUTensor]
	
	public var learningRate: Float
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], data: [GPUTensor]?) -> [GPUTensor]
	{
		return []
	}
}

public struct AdaDeltaOptimizer: Optimizer
{
	public typealias OptimizerData = [(GPUTensor, GPUTensor)]
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], data: [(GPUTensor, GPUTensor)]?) -> [(GPUTensor, GPUTensor)]
	{
		return []
	}
}


public protocol TrainingSampleProvider
{
	mutating func nextSamples(count: Int) -> [(input: GPUMatrix3, expected: GPUMatrix3)]
}


public class GPUNetworkTrainingSession<OptimizerType: Optimizer>
{
	public private(set) var network: GPUFeedForwardNeuralNetwork
	public let batchSize: Int
	public let optimizer: OptimizerType
	public var trainingSampleProvider: TrainingSampleProvider
	
	private var optimizerData: OptimizerType.OptimizerData?
	
	public init(network: GPUFeedForwardNeuralNetwork, batchSize: Int = 1, optimizer: OptimizerType, sampleProvider: TrainingSampleProvider)
	{
		self.network = network
		self.batchSize = batchSize
		self.optimizer = optimizer
		self.optimizerData = nil
		self.trainingSampleProvider = sampleProvider
	}
	
	public func train(epochs: Int)
	{
		DispatchQueue.global().async
		{ [weak self] in
			
			
			for _ in 0 ..< epochs
			{
				guard let this = self else { break }
				let samples = this.trainingSampleProvider.nextSamples(count: this.batchSize)
				
				let buffer = GPUGlobalQueue.makeCommandBuffer()
				let encoder = buffer.makeComputeCommandEncoder()
				
				for sample in samples
				{
					this.network.updateGradients(with: sample, encoder: encoder)
				}
				
				let weights = this
					.network
					.layers
					.flatMap{$0 as? GPUWeightAdjustableLayer}
					.flatMap{$0.weights}
				
				let gradients = this
					.network
					.layers
					.flatMap{$0 as? GPUWeightAdjustableLayer}
					.flatMap{$0.gradients}
				
				this.optimizerData = this.optimizer.update(weights: weights, gradients: gradients, data: this.optimizerData)
				
				encoder.endEncoding()
				buffer.commit()
				buffer.waitUntilCompleted()
			}
			
			self?.finalizeTraining()
		}
	}
	
	private func finalizeTraining()
	{
		//TODO: Copy weights back from VRAM to RAM
	}
	
	
}


