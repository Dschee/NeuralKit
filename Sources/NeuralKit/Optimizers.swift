//
//  Optimizers.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.04.17.
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
import MatrixVector


public protocol Optimizer
{
	associatedtype OptimizerData
	
	func update(weights: inout [Tensor], gradients: inout [Tensor], batchSize: Int, data: OptimizerData?) -> OptimizerData
}


public struct SGDOptimizer: Optimizer
{
	public typealias OptimizerData = Void
	
	public var learningRate: Float
	
	public init(learningRate: Float)
	{
		self.learningRate = learningRate
	}
	
	public func update(weights: inout [Tensor], gradients: inout [Tensor], batchSize: Int, data: Void?)
	{
		for index in weights.indices
		{
			weights[index].values &-= gradients[index].values &* learningRate
			gradients[index].values = zeros(gradients[index].values)
		}
	}
}

public struct MomentumOptimizer: Optimizer
{
	public typealias OptimizerData = [[Float]]
	
	public var learningRate: Float
	public var momentum: Float
	
	public init(learningRate: Float, momentum: Float)
	{
		self.learningRate = learningRate
		self.momentum = momentum
	}
	
	public func update(weights: inout [Tensor], gradients: inout [Tensor], batchSize: Int, data: [[Float]]?) -> [[Float]]
	{
		var momentumData: [[Float]]
		
		if let data = data
		{
			momentumData = data
		}
		else
		{
			momentumData = weights.map{Array(repeating: 0, count: $0.values.count)}
		}
		
		for index in weights.indices
		{
			let weightDelta = (momentumData[index] &* momentum) &+ (gradients[index].values &* learningRate)
			momentumData[index] = weightDelta
			weights[index].values &+= weightDelta
			gradients[index].values = zeros(gradients[index].values)
		}
		
		return momentumData
	}
}

public struct AdaGradOptimizer: Optimizer
{
	public typealias OptimizerData = [[Float]]
	
	public var learningRate: Float
	
	public init(learningRate: Float)
	{
		self.learningRate = learningRate
	}
	
	public func update(weights: inout [Tensor], gradients: inout [Tensor], batchSize: Int, data: [[Float]]?) -> [[Float]]
	{
		var gradientSumData: [[Float]]
		
		if let data = data
		{
			gradientSumData = data
		}
		else
		{
			gradientSumData = weights.map{Array(repeating: 0, count: $0.values.count)}
		}
		
		for index in weights.indices
		{
			gradientSumData[index] &+= gradients[index].values &* gradients[index].values
			weights[index].values &-= learningRate &/ sqrt(gradientSumData[index] &+ 1E-8) &* gradients[index].values
			gradients[index].values = zeros(gradients[index].values)
		}
		
		return gradientSumData
	}
}

public struct RMSpropOptimizer: Optimizer
{
	public typealias OptimizerData = [[Float]]
	
	public var learningRate: Float
	public var decay: Float
	
	public init(learningRate: Float, decay: Float)
	{
		self.learningRate = learningRate
		self.decay = decay
	}
	
	public func update(weights: inout [Tensor], gradients: inout [Tensor], batchSize: Int, data: [[Float]]?) -> [[Float]]
	{
		var gradientSumData: [[Float]]
		
		if let data = data
		{
			gradientSumData = data
		}
		else
		{
			gradientSumData = weights.map{Array(repeating: 0, count: $0.values.count)}
		}
		
		for index in weights.indices
		{
			gradientSumData[index] = (gradientSumData[index] &* decay) &+ (gradients[index].values &* gradients[index].values &* (1 - decay))
			weights[index].values &-= learningRate &/ sqrt(gradientSumData[index] &+ 1E-8) &* gradients[index].values
			gradients[index].values = zeros(gradients[index].values)
		}
		
		return gradientSumData
	}
}

public struct AdaDeltaOptimizer: Optimizer
{
	public typealias OptimizerData = [([Float], [Float])]
	
	public var decay: Float
	
	public init(decay: Float)
	{
		self.decay = decay
	}
	
	public func update(weights: inout [Tensor], gradients: inout [Tensor], batchSize: Int, data: [([Float], [Float])]?) -> [([Float], [Float])]
	{
		var gradientSumData: [([Float], [Float])]
		
		if let data = data
		{
			gradientSumData = data
		}
		else
		{
			gradientSumData = weights.map{Array(repeating: 0, count: $0.values.count)}.map{($0, $0)}
		}
		
		for index in weights.indices
		{
			gradientSumData[index].0 = (gradientSumData[index].0 &* decay) &+ (gradients[index].values &* gradients[index].values &* (1 - decay))
			let weightDelta = sqrt(gradientSumData[index].1 &+ 1E-8) &/ sqrt(gradientSumData[index].0 &+ 1E-8) &* gradients[index].values
			gradientSumData[index].1 = (gradientSumData[index].1 &* decay) &+ ((1 - decay) &* weightDelta &* weightDelta)
			gradients[index].values = zeros(gradients[index].values)
		}
		
		return gradientSumData
	}
}
