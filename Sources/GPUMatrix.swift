//
//  GPUMatrix.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 03.04.17.
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


public struct GPUMatrix
{
	public let descriptor: (width: UInt32, height: UInt32)
	
	private let buffer: MTLBuffer
	private let descriptorBuffer: MTLBuffer
	
	private unowned let device: MTLDevice
	
	public init(on device: MTLDevice, matrix: Matrix)
	{
		buffer = device.makeBuffer(
			bytes: matrix.values,
			length: MemoryLayout.size(ofValue: Float(0)) * matrix.values.count,
			options: []
		)
		
		descriptor = (
			width: UInt32(matrix.width),
			height: UInt32(matrix.height)
		)
		descriptorBuffer = device.makeBuffer(
			bytes: [descriptor.width, descriptor.height],
			length: MemoryLayout.size(ofValue: descriptor),
			options: []
		)
		
		self.device = device
	}
	
	private init(device: MTLDevice, descriptor: (width: UInt32, height: UInt32), buffer: MTLBuffer)
	{
		self.buffer = buffer
		
		self.descriptor = descriptor
		descriptorBuffer = device.makeBuffer(
			bytes: [descriptor.width, descriptor.height],
			length: MemoryLayout.size(ofValue: descriptor),
			options: []
		)
		
		self.device = device
	}
	
	public func asMatrix() -> Matrix
	{
		let values = Array<Float>(
			UnsafeBufferPointer(
				start: buffer.contents().assumingMemoryBound(to: Float.self),
				count: Int(descriptor.width) * Int(descriptor.height)
			)
		)
		
		return Matrix(values: values, width: Int(descriptor.width), height: Int(descriptor.height))
	}
	
	public func reshaped(width: Int, height: Int) -> GPUMatrix
	{
		precondition(
			width * height == Int(descriptor.width) * Int(descriptor.height),
			"Number of values in matrix must be equal."
		)
		
		return GPUMatrix(device: device, descriptor: (width: UInt32(width), height: UInt32(height)), buffer: buffer)
	}
	
	
	/// Sets the buffer as an argument of the encoder at the given index
	/// and sets the descriptor as the argument at index + 1.
	///
	/// - Parameters:
	///   - encoder: Encoder on which the buffer should be set as argument.
	///   - index: Index at which the buffer should be set as an argument.
	public func setBuffer(on encoder: MTLComputeCommandEncoder, at index: Int)
	{
		encoder.setBuffer(self.buffer, offset: 0, at: index)
		encoder.setBuffer(self.descriptorBuffer, offset: 0, at: index + 1)
	}
}

public struct GPUMatrix3
{
	public let descriptor: (width: UInt32, height: UInt32, depth: UInt32)
	
	private let buffer: MTLBuffer
	private let descriptorBuffer: MTLBuffer
	
	private unowned let device: MTLDevice
	
	public init(on device: MTLDevice, matrix: Matrix3)
	{
		buffer = device.makeBuffer(
			bytes: matrix.values,
			length: MemoryLayout.size(ofValue: Float(0)) * matrix.values.count,
			options: []
		)
		
		descriptor = (
			width: UInt32(matrix.width),
			height: UInt32(matrix.height),
			depth: UInt32(matrix.depth)
		)
		descriptorBuffer = device.makeBuffer(
			bytes: [descriptor.width, descriptor.height, descriptor.depth],
			length: MemoryLayout.size(ofValue: descriptor),
			options: []
		)
		
		self.device = device
	}
	
	private init(device: MTLDevice, descriptor: (width: UInt32, height: UInt32, depth: UInt32), buffer: MTLBuffer)
	{
		self.buffer = buffer
		
		self.descriptor = descriptor
		descriptorBuffer = device.makeBuffer(
			bytes: [descriptor.width, descriptor.height, descriptor.depth],
			length: MemoryLayout.size(ofValue: descriptor),
			options: []
		)
		
		self.device = device
	}
	
	public func asMatrix() -> Matrix3
	{
		let values = Array<Float>(
			UnsafeBufferPointer(
				start: buffer.contents().assumingMemoryBound(to: Float.self),
				count: Int(descriptor.width) * Int(descriptor.height) * Int(descriptor.depth)
			)
		)
		
		return Matrix3(values: values, width: Int(descriptor.width), height: Int(descriptor.height), depth: Int(descriptor.depth))
	}
	
	public func reshaped(width: Int, height: Int, depth: Int) -> GPUMatrix3
	{
		precondition(
			width * height * depth == Int(descriptor.width) * Int(descriptor.height) * Int(descriptor.depth),
			"Number of values in matrix must be equal."
		)
		
		return GPUMatrix3(device: device, descriptor: (width: UInt32(width), height: UInt32(height), depth: UInt32(depth)), buffer: buffer)
	}
	
	
	/// Sets the buffer as an argument of the encoder at the given index
	/// and sets the descriptor as the argument at index + 1.
	///
	/// - Parameters:
	///   - encoder: Encoder on which the buffer should be set as argument.
	///   - index: Index at which the buffer should be set as an argument.
	public func setBuffer(on encoder: MTLComputeCommandEncoder, at index: Int)
	{
		encoder.setBuffer(self.buffer, offset: 0, at: index)
		encoder.setBuffer(self.descriptorBuffer, offset: 0, at: index + 1)
	}
}
