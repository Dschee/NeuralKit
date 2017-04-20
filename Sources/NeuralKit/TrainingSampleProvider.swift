//
//  TrainingSampleProvider.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 20.04.17.
//
//

import Foundation


public protocol TrainingSampleProvider
{
	func next(_ count: Int) -> [TrainingSample]
}


public struct ArrayTrainingSampleProvider: TrainingSampleProvider
{
	public let samples: [TrainingSample]
	
	public init(samples: [TrainingSample])
	{
		self.samples = samples
	}
	
	public func next(_ count: Int) -> [TrainingSample]
	{
		return (0 ..< count)
			.map{_ in arc4random_uniform(UInt32(samples.count))}
			.map{Int($0)}
			.map{samples[$0]}
	}
}
