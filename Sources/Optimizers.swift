//
//  Optimizers.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.04.17.
//
//

import Foundation


public protocol Optimizer
{
	associatedtype OptimizerData
	
	func update(weights: inout [Tensor], gradients: inout [Tensor], batchSize: Int, data: OptimizerData?) -> OptimizerData
}

