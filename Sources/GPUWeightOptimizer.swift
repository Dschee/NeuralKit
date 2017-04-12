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

@available(OSX 10.12, *)
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

@available(OSX 10.12, *)
public protocol Optimizer
{
	associatedtype OptimizerData
	
	func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: OptimizerData?) -> OptimizerData
}

@available(OSX 10.12, *)
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

@available(OSX 10.12, *)
public struct MomentumOptimizer: Optimizer
{
	public typealias OptimizerData = [MTLBuffer]
	
	public var learningRate: Float
	public var momentum: Float
	
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(learningRate: Float, momentum: Float)
	{
		self.learningRate = learningRate
		self.momentum = momentum
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_momentum")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: [MTLBuffer]?) -> [MTLBuffer]
	{
		let momentumData: [MTLBuffer]
		
		if let data = data
		{
			momentumData = data
		}
		else
		{
			momentumData = weights.map
			{ (tensor) -> MTLBuffer in
				return GPUGlobalDevice.makeBuffer(
					bytes: Array<Float>(repeating: 0, count: Int(tensor.count)),
					length: Int(tensor.count) * MemoryLayout<Float>.size,
					options: .storageModePrivate
				)
			}
		}
		
		encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
		
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 4)
		encoder.setBytes([momentum], length: MemoryLayout<Float>.size, at: 5)
		
		for ((weightBuffer, gradientBuffer), momentumBuffer) in zip(zip(weights, gradients), momentumData)
		{
			encoder.setBuffer(weightBuffer.buffer, offset: 0, at: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, at: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, at: 2)
			encoder.setBuffer(momentumBuffer, offset: 0, at: 3)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
		
		return momentumData
	}
}

@available(OSX 10.12, *)
public struct AdaGradOptimizer: Optimizer
{
	public typealias OptimizerData = [MTLBuffer]
	
	public var learningRate: Float
	
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(learningRate: Float)
	{
		self.learningRate = learningRate
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_adagrad")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: [MTLBuffer]?) -> [MTLBuffer]
	{
		let squaredGradientSum: [MTLBuffer]
		
		if let data = data
		{
			squaredGradientSum = data
		}
		else
		{
			squaredGradientSum = weights.map
			{ (tensor) -> MTLBuffer in
				return GPUGlobalDevice.makeBuffer(
					bytes: Array<Float>(repeating: 0, count: Int(tensor.count)),
					length: Int(tensor.count) * MemoryLayout<Float>.size,
					options: .storageModePrivate
				)
			}
		}
		
		encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 4)
		
		for ((weightBuffer, gradientBuffer), squaredGradientSumBuffer) in zip(zip(weights, gradients), squaredGradientSum)
		{
			encoder.setBuffer(weightBuffer.buffer, offset: 0, at: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, at: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, at: 2)
			encoder.setBuffer(squaredGradientSumBuffer, offset: 0, at: 3)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
		
		return squaredGradientSum
	}
}

@available(OSX 10.12, *)
public struct AdaDeltaOptimizer: Optimizer
{
	public typealias OptimizerData = ([MTLBuffer], [MTLBuffer])
	
	public var decay: Float
	
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(decay: Float)
	{
		self.decay = decay
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_adadelta")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder, data: ([MTLBuffer], [MTLBuffer])?) -> ([MTLBuffer], [MTLBuffer])
	{
		let squaredGradientSum: [MTLBuffer]
		let squaredWeightUpdateSum: [MTLBuffer]
		
		if let data = data
		{
			(squaredGradientSum, squaredWeightUpdateSum) = data
		}
		else
		{
			squaredGradientSum = weights.map
				{ (tensor) -> MTLBuffer in
					return GPUGlobalDevice.makeBuffer(
						bytes: Array<Float>(repeating: 0, count: Int(tensor.count)),
						length: Int(tensor.count) * MemoryLayout<Float>.size,
						options: .storageModePrivate
					)
			}
			squaredWeightUpdateSum = weights.map
			{ (tensor) -> MTLBuffer in
				return GPUGlobalDevice.makeBuffer(
					bytes: Array<Float>(repeating: 0, count: Int(tensor.count)),
					length: Int(tensor.count) * MemoryLayout<Float>.size,
					options: .storageModePrivate
				)
			}
		}
		
		encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
		
		for ((weightBuffer, gradientBuffer), (squaredGradientSumBuffer, squaredWeightUpdateBuffer)) in zip(zip(weights, gradients), zip(squaredGradientSum, squaredWeightUpdateSum))
		{
			encoder.setBuffer(weightBuffer.buffer, offset: 0, at: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, at: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, at: 2)
			encoder.setBuffer(squaredGradientSumBuffer, offset: 0, at: 3)
			encoder.setBuffer(squaredWeightUpdateBuffer, offset: 0, at: 4)
			encoder.setBytes([decay], length: MemoryLayout<Float>.size, at: 5)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
		
		return (squaredGradientSum, squaredWeightUpdateSum)
	}
}


@available(OSX 10.12, *)
public protocol TrainingSampleProvider
{
	mutating func nextSamples(count: Int) -> [(input: GPUMatrix3, expected: GPUMatrix3)]
}


@available(OSX 10.12, *)
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
				autoreleasepool
				{
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
			}
			
			self?.finishTraining()
			self?.onFinishTraining?()
		}
	}
	
	private func finishTraining()
	{
		self.network.finishTraining()
	}
}

@available(OSX 10.12, *)
public struct ArrayTrainingSampleProvider: TrainingSampleProvider
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
