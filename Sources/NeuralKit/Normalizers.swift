//
//  Normalizers.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.04.17.
//
//

import Foundation
import MatrixVector


public protocol Normalizer
{
	func update(weights: [Tensor], gradients: inout [Tensor])
}


public struct L1Normalizer: Normalizer
{
	public var decay: Float
	
	public init(decay: Float)
	{
		self.decay = decay
	}
	
	public func update(weights: [Tensor], gradients: inout [Tensor])
	{
		for index in weights.indices
		{
			gradients[index].values &+= copysign(Array<Float>(repeating: decay, count: gradients[index].values.count), weights[index].values)
		}
	}
}


public struct L2Normalizer: Normalizer
{
	public var decay: Float
	
	public init(decay: Float)
	{
		self.decay = decay
	}
	
	public func update(weights: [Tensor], gradients: inout [Tensor])
	{
		for index in weights.indices
		{
			gradients[index].values &+= weights[index].values &* decay
		}
	}
}
