//
//  TrainingSampleProvider.swift
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


@available(OSX 10.12, *)
public protocol TrainingSampleProvider
{
	mutating func nextSamples(count: Int) -> [(input: GPUMatrix3, expected: GPUMatrix3)]
}


@available(OSX 10.12, *)
public struct ArrayTrainingSampleProvider: TrainingSampleProvider
{
	public let samples: [TrainingSample]
	
	public init(samples: [TrainingSample])
	{
		self.samples = samples
	}
	
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

@available(OSX 10.12, *)
public struct CachedArrayTrainingSampleProvider: TrainingSampleProvider
{
	public let samples: [TrainingSample]
	private var gpuSamples: [Int: (GPUMatrix3, GPUMatrix3)]
	
	public init(samples: [TrainingSample])
	{
		self.samples = samples
		self.gpuSamples = [:]
	}
	
	public mutating func nextSamples(count: Int) -> [(input: GPUMatrix3, expected: GPUMatrix3)]
	{
		let indices = (0 ..< count)
			.map{_ in Int(arc4random_uniform(UInt32(samples.count)))}
		
		var next:[(GPUMatrix3, GPUMatrix3)] = []
		
		for index in indices
		{
			if let sample = gpuSamples[index]
			{
				next.append(sample)
			}
			else
			{
				let sample = (
					GPUMatrix3(matrix: samples[index].values, isShared: true),
					GPUMatrix3(matrix: samples[index].expected, isShared: true)
				)
				gpuSamples[index] = sample
				next.append(sample)
				
				sample.0.buffer.addDebugMarker("Hello", range: NSRange())
			}
		}
		
		return next
	}
}
