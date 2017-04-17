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


/// A generic optimizer.
///
/// An optimizer will take a set of weights and corresponding gradients
/// and update the weights according to the gradients and some optional
/// data.
@available(OSX 10.12, *)
public protocol Optimizer
{
	
	/// Data required for optimization.
	///
	/// This data will be kept between iterations
	/// and the result of one optimization will
	/// be provided to the next optimization pass.
	associatedtype OptimizerData
	
	
	/// Updates weights based on the provided gradients and
	/// optimizer specific optimization data.
	///
	/// - Parameters:
	///   - weights: Weights, which should be updated.
	///   - gradients: Weight gradients, which correspond to the given weights.
	///   - batchSize: Number of samples which where used to calculate the given weight gradients.
	///   - encoder: Encoder for dispatching Metal kernels
	///   - data: Optimization data from the last optimization pass or nil, if this function is called for the first time.
	/// - Returns: Updated optimization data, which will be passed to the next optimization pass.
	func update(weights: [GPUTensor], gradients: [GPUTensor], batchSize: Int, encoder: MTLComputeCommandEncoder, data: OptimizerData?) -> OptimizerData
}


/// Gradient Descent Optimizer
///
/// Optimizes the weights of adjustable layers
/// using only the current gradients.
///
/// This optimizer may perform not as good as other optimizers
/// as it has a higher chance of getting stuck in a local minimum.
@available(OSX 10.12, *)
public struct SGDOptimizer: Optimizer
{
	
	/// Data required for optimization.
	///
	/// This data will be kept between iterations
	/// and the result of one optimization will
	/// be provided to the next optimization pass.
	public typealias OptimizerData = Void
	
	
	/// Rate, at which the weights will be updated
	/// according to the weight gradients.
	///
	/// A high learning rate will lead to faster convergence
	/// but can lead to inaccurate results or divergence,
	/// if it is chosen too high.
	public var learningRate: Float
	
	
	/// Optimization function state
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	
	/// Creates a new Stochastic Gradient Descent optimizer.
	///
	/// The SGDOptimizer will update the weights of adjustable layers
	/// using only the current gradients.
	///
	/// This optimizer may perform not as good as other optimizers
	/// as it has a higher chance of getting stuck in a local minimum.
	///
	/// - Parameter learningRate: Rate, at which weight will be adapted.
	public init(learningRate: Float)
	{
		self.learningRate = learningRate
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_sgd")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	
	/// Updates weights based on the provided gradients and
	/// optimizer specific optimization data.
	///
	/// - Parameters:
	///   - weights: Weights, which should be updated.
	///   - gradients: Weight gradients, which correspond to the given weights.
	///   - batchSize: Number of samples which where used to calculate the given weight gradients.
	///   - encoder: Encoder for dispatching Metal kernels
	///   - data: Optimization data from the last optimization pass or nil, if this function is called for the first time.
	/// - Returns: Updated optimization data, which will be passed to the next optimization pass.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], batchSize: Int, encoder: MTLComputeCommandEncoder, data: Void?) -> Void
	{
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 3)
		encoder.setBytes([Float(batchSize)], length: MemoryLayout<Float>.size, at: 4)
		
		for (weightBuffer, gradientBuffer) in zip(weights, gradients)
		{
			encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
			
			encoder.setBuffer(weightBuffer.buffer, offset: 0, at: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, at: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, at: 2)
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
	}
	
}


/// Gradient Descent Optimizer with momentum.
///
/// This optimizer extends the standard SGDOptimizer
/// by storing exponentially decaying previous weight updates,
/// which are added to the weight update.
///
/// A momentum optimizer can lead to better results than a SGDOptimizer
/// as the momentum term can help overcome local minima.
@available(OSX 10.12, *)
public struct MomentumOptimizer: Optimizer
{
	
	/// Data required for optimization.
	///
	/// This data will be kept between iterations
	/// and the result of one optimization will
	/// be provided to the next optimization pass.
	public typealias OptimizerData = [MTLBuffer]
	
	
	/// Rate, at which the weights will be updated
	/// according to the weight gradients.
	///
	/// A high learning rate will lead to faster convergence
	/// but can lead to inaccurate results or divergence,
	/// if it is chosen too high.
	public var learningRate: Float
	
	
	/// Decay rate of previous weight updates.
	///
	/// If the momentum is chosen hight, previous weight updates
	/// will decay slower and have a higher importance on 
	/// subsequent weight updates.
	///
	/// This value must be less than 1 but bigger than 0.
	/// If the momentum value is zero, the momentum optimizer
	/// will be equivalent to a SGDOptimizer.
	public var momentum: Float
	
	
	/// GPU optimize function state
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	
	/// Creates a new momentum optimizer.
	///
	/// This optimizer extends the standard SGDOptimizer
	/// by storing exponentially decaying previous weight updates,
	/// which are added to the weight update.
	///
	/// A momentum optimizer can lead to better results than a SGDOptimizer
	/// as the momentum term can help overcome local minima.
	///
	/// - Parameters:
	///   - learningRate: Rate, at which weights should be adapted to weight gradients.
	///   - momentum: Decay rate of previous weight updates. This value should be around 0.8. 
	///			It must not be greater than or equal to 1.
	public init(learningRate: Float, momentum: Float)
	{
		self.learningRate = learningRate
		self.momentum = momentum
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_momentum")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	
	/// Updates weights based on the provided gradients and
	/// optimizer specific optimization data.
	///
	/// - Parameters:
	///   - weights: Weights, which should be updated.
	///   - gradients: Weight gradients, which correspond to the given weights.
	///   - batchSize: Number of samples which where used to calculate the given weight gradients.
	///   - encoder: Encoder for dispatching Metal kernels
	///   - data: Optimization data from the last optimization pass or nil, if this function is called for the first time.
	/// - Returns: Updated optimization data, which will be passed to the next optimization pass.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], batchSize: Int, encoder: MTLComputeCommandEncoder, data: [MTLBuffer]?) -> [MTLBuffer]
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
		
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 4)
		encoder.setBytes([momentum], length: MemoryLayout<Float>.size, at: 5)
		encoder.setBytes([Float(batchSize)], length: MemoryLayout<Float>.size, at: 6)
		
		for ((weightBuffer, gradientBuffer), momentumBuffer) in zip(zip(weights, gradients), momentumData)
		{
			encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
			
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
public struct NesterovOptimizer: Optimizer
{
	
	/// Data required for optimization.
	///
	/// This data will be kept between iterations
	/// and the result of one optimization will
	/// be provided to the next optimization pass.
	public typealias OptimizerData = [MTLBuffer]
	
	public var learningRate: Float
	public var momentum: Float
	
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(learningRate: Float, momentum: Float)
	{
		self.learningRate = learningRate
		self.momentum = momentum
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_nesterov")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	
	/// Updates weights based on the provided gradients and
	/// optimizer specific optimization data.
	///
	/// - Parameters:
	///   - weights: Weights, which should be updated.
	///   - gradients: Weight gradients, which correspond to the given weights.
	///   - batchSize: Number of samples which where used to calculate the given weight gradients.
	///   - encoder: Encoder for dispatching Metal kernels
	///   - data: Optimization data from the last optimization pass or nil, if this function is called for the first time.
	/// - Returns: Updated optimization data, which will be passed to the next optimization pass.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], batchSize: Int, encoder: MTLComputeCommandEncoder, data: [MTLBuffer]?) -> [MTLBuffer]
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
		
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 4)
		encoder.setBytes([momentum], length: MemoryLayout<Float>.size, at: 5)
		encoder.setBytes([Float(batchSize)], length: MemoryLayout<Float>.size, at: 6)
		
		for ((weightBuffer, gradientBuffer), momentumBuffer) in zip(zip(weights, gradients), momentumData)
		{
			encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
			
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
	
	/// Data required for optimization.
	///
	/// This data will be kept between iterations
	/// and the result of one optimization will
	/// be provided to the next optimization pass.
	public typealias OptimizerData = [MTLBuffer]
	
	public var learningRate: Float
	
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(learningRate: Float)
	{
		self.learningRate = learningRate
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_adagrad")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	
	/// Updates weights based on the provided gradients and
	/// optimizer specific optimization data.
	///
	/// - Parameters:
	///   - weights: Weights, which should be updated.
	///   - gradients: Weight gradients, which correspond to the given weights.
	///   - batchSize: Number of samples which where used to calculate the given weight gradients.
	///   - encoder: Encoder for dispatching Metal kernels
	///   - data: Optimization data from the last optimization pass or nil, if this function is called for the first time.
	/// - Returns: Updated optimization data, which will be passed to the next optimization pass.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], batchSize: Int, encoder: MTLComputeCommandEncoder, data: [MTLBuffer]?) -> [MTLBuffer]
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
		
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 4)
		encoder.setBytes([Float(batchSize)], length: MemoryLayout<Float>.size, at: 5)
		
		for ((weightBuffer, gradientBuffer), squaredGradientSumBuffer) in zip(zip(weights, gradients), squaredGradientSum)
		{
			encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
		
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
public struct RMSpropOptimizer: Optimizer
{
	
	/// Data required for optimization.
	///
	/// This data will be kept between iterations
	/// and the result of one optimization will
	/// be provided to the next optimization pass.
	public typealias OptimizerData = [MTLBuffer]
	
	public var learningRate: Float
	public var decay: Float
	
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(learningRate: Float, decay: Float)
	{
		self.learningRate = learningRate
		self.decay = decay
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_rmsprop")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	
	/// Updates weights based on the provided gradients and
	/// optimizer specific optimization data.
	///
	/// - Parameters:
	///   - weights: Weights, which should be updated.
	///   - gradients: Weight gradients, which correspond to the given weights.
	///   - batchSize: Number of samples which where used to calculate the given weight gradients.
	///   - encoder: Encoder for dispatching Metal kernels
	///   - data: Optimization data from the last optimization pass or nil, if this function is called for the first time.
	/// - Returns: Updated optimization data, which will be passed to the next optimization pass.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], batchSize: Int, encoder: MTLComputeCommandEncoder, data: [MTLBuffer]?) -> [MTLBuffer]
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
		
		encoder.setBytes([learningRate], length: MemoryLayout<Float>.size, at: 4)
		encoder.setBytes([decay], length: MemoryLayout<Float>.size, at: 5)
		encoder.setBytes([Float(batchSize)], length: MemoryLayout<Float>.size, at: 6)
		
		for ((weightBuffer, gradientBuffer), squaredGradientSumBuffer) in zip(zip(weights, gradients), squaredGradientSum)
		{
			encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
			
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
	
	/// Data required for optimization.
	///
	/// This data will be kept between iterations
	/// and the result of one optimization will
	/// be provided to the next optimization pass.
	public typealias OptimizerData = ([MTLBuffer], [MTLBuffer])
	
	public var decay: Float
	
	private let optimizeFunctionPipelineState: MTLComputePipelineState
	
	public init(decay: Float)
	{
		self.decay = decay
		
		let function = GPUGlobalLibrary.makeFunction(name: "Optimize_adadelta")!
		self.optimizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: function)
	}
	
	
	/// Updates weights based on the provided gradients and
	/// optimizer specific optimization data.
	///
	/// - Parameters:
	///   - weights: Weights, which should be updated.
	///   - gradients: Weight gradients, which correspond to the given weights.
	///   - batchSize: Number of samples which where used to calculate the given weight gradients.
	///   - encoder: Encoder for dispatching Metal kernels
	///   - data: Optimization data from the last optimization pass or nil, if this function is called for the first time.
	/// - Returns: Updated optimization data, which will be passed to the next optimization pass.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], batchSize: Int, encoder: MTLComputeCommandEncoder, data: ([MTLBuffer], [MTLBuffer])?) -> ([MTLBuffer], [MTLBuffer])
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
		
		encoder.setBytes([decay], length: MemoryLayout<Float>.size, at: 5)
		encoder.setBytes([Float(batchSize)], length: MemoryLayout<Float>.size, at: 6)
		
		for ((weightBuffer, gradientBuffer), (squaredGradientSumBuffer, squaredWeightUpdateBuffer)) in zip(zip(weights, gradients), zip(squaredGradientSum, squaredWeightUpdateSum))
		{
			encoder.setComputePipelineState(self.optimizeFunctionPipelineState)
			
			encoder.setBuffer(weightBuffer.buffer, offset: 0, at: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, at: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, at: 2)
			encoder.setBuffer(squaredGradientSumBuffer, offset: 0, at: 3)
			encoder.setBuffer(squaredWeightUpdateBuffer, offset: 0, at: 4)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
		
		return (squaredGradientSum, squaredWeightUpdateSum)
	}
}
