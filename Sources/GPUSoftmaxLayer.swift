//
//  GPUSoftmaxLayer.swift
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


public struct GPUSoftmaxLayer: GPUOutputLayer
{
	public let inputSize: (width: Int, height: Int, depth: Int)
	public var outputSize: (width: Int, height: Int, depth: Int)
	{
		return inputSize
	}
	
	private var gpuFunctionPipelineState: MTLComputePipelineState!
	private var gpuExpFunctionPipelineState: MTLComputePipelineState!
	private var gpuLossPipelineState: MTLComputePipelineState!
	private var gpuExponentiated: GPUMatrix3!
	private var gpuOutput: GPUMatrix3!
	private var gpuGradient: GPUMatrix3!
	
	public init(inputSize: (width: Int, height: Int, depth: Int))
	{
		self.inputSize = inputSize
	}
	
	public mutating func initialize(library: MTLLibrary, shareOutput: Bool)
	{
		guard
			let exponentiate = library.makeFunction(name: "SoftmaxLayer_forward_exp"),
			let function = library.makeFunction(name: "SoftmaxLayer_forward"),
			let loss = library.makeFunction(name: "Loss_delta")
			else
		{
			fatalError()
		}
		
		do
		{
			self.gpuExpFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: exponentiate)
			self.gpuFunctionPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: function)
			self.gpuLossPipelineState = try GPUGlobalDevice.makeComputePipelineState(function: loss)
		}
		catch
		{
			fatalError("\(error)")
		}
		
		let exponentiatedMatrix = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		self.gpuExponentiated = GPUMatrix3(matrix: exponentiatedMatrix)
		
		let outputMatrix = Matrix3(repeating: 0, width: outputSize.width, height: outputSize.height, depth: outputSize.depth)
		self.gpuOutput = GPUMatrix3(matrix: outputMatrix, isShared: shareOutput)
		
		let gradientMatrix = Matrix3(repeating: 0, width: inputSize.width, height: inputSize.height, depth: inputSize.depth)
		self.gpuGradient = GPUMatrix3(matrix: gradientMatrix, isShared: shareOutput)
	}
	
	public func forward(_ input: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(gpuExpFunctionPipelineState)
		
		input.setBuffer(on: encoder, at: 0)
		gpuExponentiated.setBuffer(on: encoder, at: 2)
		
		encoder.dispatch(workSize: outputSize)
		
		encoder.setComputePipelineState(gpuFunctionPipelineState)
		
		gpuExponentiated.setBuffer(on: encoder, at: 0)
		gpuOutput.setBuffer(on: encoder, at: 2)
		
		encoder.dispatch(workSize: outputSize)
		
		return gpuOutput
	}
	
	public func loss(expected: GPUMatrix3, actual: GPUMatrix3, encoder: MTLComputeCommandEncoder) -> GPUMatrix3
	{
		encoder.setComputePipelineState(gpuLossPipelineState)
		
		// When using the cross entropy loss function for softmax activation, 
		// the term simplifies to actual - expected (opposite of euclidean distance loss for non softmax functions) 
		expected.setBuffer(on: encoder, at: 2)
		actual.setBuffer(on: encoder, at: 0)
		gpuGradient.setBuffer(on: encoder, at: 4)
		
		encoder.dispatch(workSize: outputSize)
		
		return gpuGradient
	}
}


