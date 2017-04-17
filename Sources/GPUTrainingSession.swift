//
//  GPUTrainingSession.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 13.04.17.
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
public class GPUNetworkTrainingSession<OptimizerType: Optimizer>
{
	public private(set) var network: GPUFeedForwardNeuralNetwork
	public let batchSize: Int
	
	public let optimizer: OptimizerType
	public let normalizers: [Normalizer]
	
	public var trainingSampleProvider: TrainingSampleProvider
	
	public var onBatchFinish: ((_ loss: Float, _ epoch: Int) -> ())?
	public var onFinishTraining: (() -> ())?
	
	private var optimizerData: OptimizerType.OptimizerData?
	
	public private(set) var isTraining: Bool = false
	
	
	deinit
	{
		if self.isTraining
		{
			print("Warning: Training session deallocated while still training. Interrupting training")
		}
		print("Training session deallocated")
	}
	
	public init(network: GPUFeedForwardNeuralNetwork, batchSize: Int = 1, optimizer: OptimizerType, normalizers: [Normalizer] = [], sampleProvider: TrainingSampleProvider)
	{
		self.network = network
		self.batchSize = batchSize
		self.optimizer = optimizer
		self.normalizers = normalizers
		self.optimizerData = nil
		self.trainingSampleProvider = sampleProvider
	}
	
	public func train(epochs: Int)
	{
		guard !isTraining else
		{
			print("Warning: Training session is already running. Cannot start again.")
			return
		}
		
		isTraining = true
		
		DispatchQueue.global().async
		{ [weak self] in
			
			for epoch in 0 ..< epochs
			{
				guard let this = self else { break }
				
				autoreleasepool // required, otherwise unbounded memory growth
				{
					let batch = this.trainingSampleProvider.nextSamples(count: this.batchSize)
					
					let buffer = GPUGlobalQueue.makeCommandBuffer()
					let encoder = buffer.makeComputeCommandEncoder()
					
					for sample in batch
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
						.flatMap{$0.weightGradients}
					
					
					for normalizer in this.normalizers
					{
						normalizer.update(weights: weights, gradients: gradients, encoder: encoder)
					}
					
					this.optimizerData = this.optimizer.update(weights: weights, gradients: gradients, batchSize: this.batchSize, encoder: encoder, data: this.optimizerData)
					
					encoder.endEncoding()
					buffer.commit()
					buffer.waitUntilCompleted()
					
					self?.onBatchFinish?(0, epoch)
				}
			}
			
			print("Finished training.")
			
			self?.finishTraining()
			self?.onFinishTraining?()
			
			self?.isTraining = false
		}
	}
	
	private func finishTraining()
	{
		self.network.finishTraining()
	}
}
