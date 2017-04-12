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
	
	internal var count: UInt32
	{
		switch self
		{
		case .vector(_, length: let length):
			return UInt32(length)
			
		case .matrix(let matrix):
			let descriptor = matrix.descriptor
			return descriptor.width * descriptor.height
			
		case .matrix3(let matrix):
			let descriptor = matrix.descriptor
			return descriptor.width * descriptor.height * descriptor.depth
		}
	}
	
	internal var buffer: MTLBuffer
	{
		switch self
		{
		case .vector(let buffer, length: _):
			return buffer
			
		case .matrix(let matrix):
			return matrix.buffer
		
		case .matrix3(let matrix):
			return matrix.buffer
		}
	}
}

public protocol Optimizer
{
	associatedtype OptimizerData
	
	func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: OptimizerData?) -> OptimizerData
}

public struct SGDOptimizer: Optimizer
{
	public typealias OptimizerData = Void
	
	public var learningRate: Float
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(learningRate: Float)
	{
		self.learningRate = learningRate
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_sgd")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: Void?) -> Void
	{
		encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
		
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 3)
		
		for (weightBuffer, gradientBuffer) in zip(weights, gradients)
		{
			encoder.setBuffer(weightBuffer.buffer, offset: 0, at: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, at: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, at: 2)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
	}
}

public struct MomentumOptimizer: Optimizer
{
	public typealias OptimizerData = [GPUTensor]
	
	public var learningRate: Float
	public var momentum: Float
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: [GPUTensor]?) -> [GPUTensor]
	{
		return []
	}
}

public struct AdaGradOptimizer: Optimizer
{
	public typealias OptimizerData = [GPUTensor]
	
	public var learningRate: Float
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: [GPUTensor]?) -> [GPUTensor]
	{
		return []
	}
}

public struct AdaDeltaOptimizer: Optimizer
{
	public typealias OptimizerData = [(GPUTensor, GPUTensor)]
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: [(GPUTensor, GPUTensor)]?) -> [(GPUTensor, GPUTensor)]
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
	
	public var onBatchFinish: ((_ loss: Float, _ epoch: Int) -> ())?
	public var onFinishTraining: (() -> ())?
	
	
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
			
			for epoch in 0 ..< epochs
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
				
				this.optimizerData = this.optimizer.update(weights: weights, gradients: gradients, encoder: encoder, data: this.optimizerData)
				
				encoder.endEncoding()
				buffer.commit()
				buffer.waitUntilCompleted()
				
				self?.onBatchFinish?(0, epoch)
			}
			
			self?.finalizeTraining()
			self?.onFinishTraining?()
		}
	}
	
	private func finalizeTraining()
	{
		//TODO: Copy weights back from VRAM to RAM
	}
	
	
}

public struct BufferedTrainingSampleProvider: TrainingSampleProvider
{
	public let samples: [TrainingSample]
	
	public func nextSamples(count: Int) -> [(input: GPUMatrix3, expected: GPUMatrix3)]
	{
		return (0 ..< count)
			.map{_ in Int(arc4random_uniform(UInt32(samples.count)))}
			.map{samples[$0]}
			.map{(
				input: GPUMatrix3(matrix: $0.values, isShared: true),
				expected: GPUMatrix3(matrix: $0.expected, isShared: true)
			)}
	}
}
