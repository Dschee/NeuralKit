//
//  GPUNormalizers.swift
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
public protocol Normalizer
{
	func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder)
}


@available(OSX 10.12, *)
public struct L1Normalizer: Normalizer
{
	public var decay: Float
	
	private let normalizeFunctionPipelineState: MTLComputePipelineState
	
	public init(decay: Float)
	{
		self.decay = decay
		
		let normalizeFunction = GPUGlobalLibrary.makeFunction(name: "Normalize_l1")!
		self.normalizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: normalizeFunction)
	}
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder)
	{
		encoder.setComputePipelineState(normalizeFunctionPipelineState)
		encoder.setBytes([decay], length: MemoryLayout<Float>.size, at: 3)
		
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
public struct L2Normalizer: Normalizer
{
	public var decay: Float
	
	private let normalizeFunctionPipelineState: MTLComputePipelineState
	
	public init(decay: Float)
	{
		self.decay = decay
		
		let normalizeFunction = GPUGlobalLibrary.makeFunction(name: "Normalize_l2")!
		self.normalizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: normalizeFunction)
	}
	
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder)
	{
		encoder.setComputePipelineState(normalizeFunctionPipelineState)
		encoder.setBytes([decay], length: MemoryLayout<Float>.size, at: 3)
		
		for (weightBuffer, gradientBuffer) in zip(weights, gradients)
		{
			encoder.setBuffer(weightBuffer.buffer, offset: 0, at: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, at: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, at: 2)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
	}
}
