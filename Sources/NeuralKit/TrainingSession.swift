//
//  TrainingSession.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 18.04.17.
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






public class NetworkTrainingSession<OptimizerType: Optimizer>
{
	public private(set) var network: FeedForwardNeuralNetwork
	
	public let batchSize: Int
	
	public var optimizer: OptimizerType
	public var normalizers: [Normalizer]
	
	public var trainingSampleProvider: TrainingSampleProvider
	
	public var onBatchFinish: ((_ loss: Float, _ epoch: Int) -> ())?
	public var onFinishTraining: (() -> ())?
	
	public private(set) var isTraining: Bool = false
	
	private var optimizerData: [OptimizerType.OptimizerData?]
	
	deinit
	{
		if self.isTraining
		{
			print("Warning: Training session deallocated while still training. Interrupting Training. To prevent this, keep at least one strong reference to a training session.")
		}
	}
	
	public init(network: FeedForwardNeuralNetwork, batchSize: Int, optimizer: OptimizerType, normalizers: [Normalizer], sampleProvider: TrainingSampleProvider)
	{
		self.network = network
		self.optimizer = optimizer
		self.normalizers = normalizers
		self.batchSize = batchSize
		self.trainingSampleProvider = sampleProvider
		self.optimizerData = Array(repeating: nil, count: network.layers.count)
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
				
				let batch = this.trainingSampleProvider.next(this.batchSize)
				
				for sample in batch
				{
					this.network.backpropagate(sample)
				}
				
				for index in this.network.layers.indices
				{
					guard var layer = this.network.layers[index] as? WeightAdjustableLayer else { continue }
					
					for normalizer in this.normalizers
					{
						normalizer.update(weights: layer.weights, gradients: &layer.weightGradients)
					}
					
					var updatedLayer = layer
					
					this.optimizerData[index] = this.optimizer.update(
						weights: &updatedLayer.weights,
						gradients: &layer.weightGradients,
						batchSize: this.batchSize,
						data: this.optimizerData[index]
					)
					
					this.network.layers[index] = updatedLayer
				}
				
				this.onBatchFinish?(0, epoch)
			}
			
			self?.isTraining = false
			self?.onFinishTraining?()
		}
	}
}
