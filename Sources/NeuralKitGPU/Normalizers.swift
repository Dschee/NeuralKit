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

/// Normalization Protocol
///
/// A normalizer introduces another loss term to a weight gradient matrix
/// which penalizes big weight values and thereby keeping weights from getting very big.
@available(OSX 10.12, *)
public protocol GPUNormalizer
{
	
	/// Updates the weight gradients based on the values in the weight matrix.
	///
	/// This will lead to weights getting smaller in a subsequent optimization pass.
	///
	/// - Parameters:
	///   - weights: Weights for which normalization loss should be calculated.
	///   - gradients: Weight gradients corresponding to the given weights.
	///   - encoder: Encoder for dispatching on GPU.
	func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder)
}


/// A L1Normalizer introduces linear loss proportional to the values in the weight matrix.
///
/// An L1 normalizer will make all weights decay equally fast towards zero.
@available(OSX 10.12, *)
public struct L1Normalizer: GPUNormalizer
{
	
	/// Weight decay rate
	///
	/// The weight decay rate determines the magnitude
	/// with which big weights should be penalized.
	public var decay: Float
	
	
	/// GPU normalization function state.
	private let normalizeFunctionPipelineState: MTLComputePipelineState
	
	
	/// Initializes a L1 normalizer.
	///
	/// A L1 normalizer introduces linear loss proportional to the values in the weight matrix.
	/// This will make all weights decay equally fast towards zero.
	///
	/// - Parameter decay: Weight decay rate
	public init(decay: Float)
	{
		self.decay = decay
		
		let normalizeFunction = GPUGlobalLibrary.makeFunction(name: "Normalize_l1")!
		self.normalizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: normalizeFunction)
	}
	
	
	/// Updates the weight gradients based on the values in the weight matrix.
	///
	/// This will lead to weights getting smaller in a subsequent optimization pass.
	///
	/// - Parameters:
	///   - weights: Weights for which normalization loss should be calculated.
	///   - gradients: Weight gradients corresponding to the given weights.
	///   - encoder: Encoder for dispatching on GPU.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder)
	{
		encoder.setComputePipelineState(normalizeFunctionPipelineState)
		encoder.setBytes([decay], length: MemoryLayout<Float>.size, index: 3)
		
		for (weightBuffer, gradientBuffer) in zip(weights, gradients)
		{
			encoder.setBuffer(weightBuffer.buffer, offset: 0, index: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, index: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, index: 2)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
	}
	
}


/// A L2 normalizer introduces quadratic loss proportional to the values in the weight matrix.
///
/// This will lead to faster decay for big weights and slower decay for small weights.
@available(OSX 10.12, *)
public struct L2Normalizer: GPUNormalizer
{
	
	/// Weight decay rate
	///
	/// The weight decay rate determines the magnitude
	/// with which big weights should be penalized.
	public var decay: Float
	
	
	/// GPU normalization function state
	private let normalizeFunctionPipelineState: MTLComputePipelineState
	
	
	/// Initializes a L2 normalizer.
	///
	/// A L1Normalizer introduces quadratic loss proportional to the values in the weight matrix.
	/// This will lead to faster decay for big weights and slower decay for small weights.
	///
	/// - Parameter decay: Weight decay rate
	public init(decay: Float)
	{
		self.decay = decay
		
		let normalizeFunction = GPUGlobalLibrary.makeFunction(name: "Normalize_l2")!
		self.normalizeFunctionPipelineState = try! GPUGlobalDevice.makeComputePipelineState(function: normalizeFunction)
	}
	
	
	/// Updates the weight gradients based on the values in the weight matrix.
	///
	/// This will lead to weights getting smaller in a subsequent optimization pass.
	///
	/// - Parameters:
	///   - weights: Weights for which normalization loss should be calculated.
	///   - gradients: Weight gradients corresponding to the given weights.
	///   - encoder: Encoder for dispatching on GPU.
	public func update(weights: [GPUTensor], gradients: [GPUTensor], encoder: MTLComputeCommandEncoder)
	{
		encoder.setComputePipelineState(normalizeFunctionPipelineState)
		encoder.setBytes([decay], length: MemoryLayout<Float>.size, index: 3)
		
		for (weightBuffer, gradientBuffer) in zip(weights, gradients)
		{
			encoder.setBuffer(weightBuffer.buffer, offset: 0, index: 0)
			encoder.setBytes([weightBuffer.count], length: MemoryLayout<UInt32>.size, index: 1)
			encoder.setBuffer(gradientBuffer.buffer, offset: 0, index: 2)
			
			encoder.dispatch(workSize: (width: Int(weightBuffer.count), height: 1, depth: 1))
		}
	}
}
