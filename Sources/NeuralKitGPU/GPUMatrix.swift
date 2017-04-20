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

/// A two dimensional Matrix which is accessible to the GPU.
/// 
///
@available(OSX 10.12, *)
public struct GPUMatrix
{
	
	/// Describes the size of the Matrix.
	public let descriptor: (width: UInt32, height: UInt32)
	
	
	/// The buffer holding the actual matrix values on the GPU.
	internal let buffer: MTLBuffer
	
	
	/// The buffer holding the descriptor on the GPU.
	internal let descriptorBuffer: MTLBuffer
	
	
	/// Creates a new two dimensional Matrix which is accessible to the GPU
	/// and copies the content of the given matrix into it.
	///
	/// If the matrix is not shared, it will be allocated in VRAM,
	/// which improves access performance from shader code
	/// but introduces a slow copy back phase if the CPU
	/// needs to access the content of the matrix.
	///
	/// - Parameters:
	///   - matrix: The matrix which should be made available to the GPU
	///   - isShared: If the matrix is shared, it will be allocated in shared memory.
	///					Otherwise it will be allocated on GPU memory.
	public init(matrix: Matrix, isShared: Bool = false)
	{
		buffer = GPUGlobalDevice.makeBuffer(
			bytes: matrix.values,
			length: MemoryLayout.size(ofValue: Float(0)) * matrix.values.count,
			options: isShared ? .storageModeShared : .storageModePrivate
		)
		
		descriptor = (
			width: UInt32(matrix.width),
			height: UInt32(matrix.height)
		)
		descriptorBuffer = GPUGlobalDevice.makeBuffer(
			bytes: [descriptor.width, descriptor.height],
			length: MemoryLayout.size(ofValue: descriptor),
			options: .storageModeShared
		)
	}
	
	
	/// Initializes a new matrix from a size descriptor and an already
	/// existing buffer.
	///
	/// - Parameters:
	///   - descriptor: Describes the size of the matrix.
	///   - buffer: Buffer storing the contents of the matrix.
	internal init(descriptor: (width: UInt32, height: UInt32), buffer: MTLBuffer)
	{
		self.buffer = buffer
		
		self.descriptor = descriptor
		descriptorBuffer = GPUGlobalDevice.makeBuffer(
			bytes: [descriptor.width, descriptor.height],
			length: MemoryLayout.size(ofValue: descriptor),
			options: .storageModeShared
		)
	}
	
	
	/// Copies the content of the gpu matrix into a new matrix.
	///
	/// If the matrix is not shared, this requires a copy from 
	/// VRAM to RAM which can decrease performance.
	///
	/// - Returns: A copy of the data contained in the gpu matrix.
	public func asMatrix() -> Matrix
	{
		
		let destination: MTLBuffer
		
		if self.buffer.storageMode == .private
		{
			destination = GPUGlobalDevice.makeBuffer(
				length: MemoryLayout<Float>.size * Int(descriptor.width) * Int(descriptor.height),
				options: .storageModeShared
			)
			
			let buffer = GPUGlobalQueue.makeCommandBuffer()
			let encoder = buffer.makeBlitCommandEncoder()
			encoder.copy(
				from: self.buffer,
				sourceOffset: 0,
				to: destination,
				destinationOffset: 0,
				size: Int(descriptor.width) * Int(descriptor.height) * MemoryLayout<Float>.size
			)
			encoder.endEncoding()
			buffer.commit()
			buffer.waitUntilCompleted()
		}
		else
		{
			destination = self.buffer
		}
		
		
		let values = Array<Float>(
			UnsafeBufferPointer(
				start: destination.contents().assumingMemoryBound(to: Float.self),
				count: Int(descriptor.width) * Int(descriptor.height)
			)
		)
		
		return Matrix(values: values, width: Int(descriptor.width), height: Int(descriptor.height))
	}
	
	
	/// Reshapes the matrix to the given width and height.
	///
	/// The product of the new width and height must equal
	/// the product of the old width and height.
	///
	/// The contents of the matrix are not copied so a write access
	/// to either the current or the reshaped matrix will result in
	/// the same changes to the other matrix.
	///
	/// - Parameters:
	///   - width: Width of the reshaped matrix
	///   - height: Height of the reshaped matrix
	/// - Returns: Reshaped matrix containing the same values
	public func reshaped(width: Int, height: Int) -> GPUMatrix
	{
		precondition(
			width * height == Int(descriptor.width) * Int(descriptor.height),
			"Number of values in matrix must be equal."
		)
		
		return GPUMatrix(descriptor: (width: UInt32(width), height: UInt32(height)), buffer: buffer)
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


/// A three dimensional matrix which is accessible to the GPU.
@available(OSX 10.12, *)
public struct GPUMatrix3
{
	
	/// Describes the width, height and depth of the matrix.
	public let descriptor: (width: UInt32, height: UInt32, depth: UInt32)
	
	
	/// The buffer holding the actual matrix values on the GPU.
	internal let buffer: MTLBuffer
	
	
	/// The buffer holding the descriptor on the GPU.
	internal let descriptorBuffer: MTLBuffer
	
	
	/// Creates a new three dimensional Matrix which is accessible to the GPU
	/// and copies the content of the given matrix into it.
	///
	/// If the matrix is not shared, it will be allocated in VRAM,
	/// which improves access performance from shader code
	/// but introduces a slow copy back phase if the CPU
	/// needs to access the content of the matrix.
	///
	/// - Parameters:
	///   - matrix: The matrix which should be made available to the GPU
	///   - isShared: If the matrix is shared, it will be allocated in shared memory.
	///					Otherwise it will be allocated on GPU memory.
	public init(matrix: Matrix3, isShared: Bool = false)
	{
		buffer = GPUGlobalDevice.makeBuffer(
			bytes: matrix.values,
			length: MemoryLayout.size(ofValue: Float(0)) * matrix.values.count,
			options: isShared ? .storageModeShared : .storageModePrivate
		)
		
		descriptor = (
			width: UInt32(matrix.width),
			height: UInt32(matrix.height),
			depth: UInt32(matrix.depth)
		)
		descriptorBuffer = GPUGlobalDevice.makeBuffer(
			bytes: [descriptor.width, descriptor.height, descriptor.depth],
			length: MemoryLayout.size(ofValue: descriptor),
			options: .storageModeShared
		)
	}
	
	
	/// Initializes a new matrix from a size descriptor and an already
	/// existing buffer.
	///
	/// - Parameters:
	///   - descriptor: Describes the size of the matrix.
	///   - buffer: Buffer storing the contents of the matrix.
	private init(descriptor: (width: UInt32, height: UInt32, depth: UInt32), buffer: MTLBuffer)
	{
		self.buffer = buffer
		
		self.descriptor = descriptor
		descriptorBuffer = GPUGlobalDevice.makeBuffer(
			bytes: [descriptor.width, descriptor.height, descriptor.depth],
			length: MemoryLayout.size(ofValue: descriptor),
			options: .storageModeShared
		)
	}
	
	
	/// Initializes a new matrix from a size descriptor,
	/// an already existing descriptor buffer and value buffer.
	///
	/// - Parameters:
	///   - descriptor: Describes the size of the matrix.
	///   - buffer: Buffer storing the contents of the matrix.
	internal init(descriptor: (width: UInt32, height: UInt32, depth: UInt32), buffer: MTLBuffer, descriptorBuffer: MTLBuffer)
	{
		self.buffer = buffer
		
		self.descriptor = descriptor
		self.descriptorBuffer = descriptorBuffer
	}
	
	
	/// Copies the content of the gpu matrix into a new matrix.
	///
	/// If the matrix is not shared, this requires a copy from
	/// VRAM to RAM which can decrease performance.
	///
	/// - Returns: A copy of the data contained in the gpu matrix.
	public func asMatrix() -> Matrix3
	{
		let destination: MTLBuffer
		
		if self.buffer.storageMode == .private
		{
			destination = GPUGlobalDevice.makeBuffer(
				length: MemoryLayout<Float>.size * Int(descriptor.width) * Int(descriptor.height) * Int(descriptor.depth),
				options: .storageModeShared
			)
			
			let buffer = GPUGlobalQueue.makeCommandBuffer()
			let encoder = buffer.makeBlitCommandEncoder()
			encoder.copy(
				from: self.buffer,
				sourceOffset: 0,
				to: destination,
				destinationOffset: 0,
				size: Int(descriptor.width) * Int(descriptor.height) * Int(descriptor.depth) * MemoryLayout<Float>.size
			)
			encoder.endEncoding()
			buffer.commit()
			buffer.waitUntilCompleted()
		}
		else
		{
			destination = self.buffer
		}
		
		let values = Array<Float>(
			UnsafeBufferPointer(
				start: destination.contents().assumingMemoryBound(to: Float.self),
				count: Int(descriptor.width) * Int(descriptor.height) * Int(descriptor.depth)
			)
		)
		
		return Matrix3(values: values, width: Int(descriptor.width), height: Int(descriptor.height), depth: Int(descriptor.depth))
	}
	
	
	/// Reshapes the matrix to the given width and height.
	///
	/// The product of the new width, height and depth must equal
	/// the product of the old width, height and depth.
	///
	/// The contents of the matrix are not copied so a write access
	/// to either the current or the reshaped matrix will result in
	/// the same changes to the other matrix.
	///
	/// - Parameters:
	///   - width: Width of the reshaped matrix
	///   - height: Height of the reshaped matrix
	///   - depth: Depth of the reshaped matrix
	/// - Returns: Reshaped matrix containing the same values
	public func reshaped(width: Int, height: Int, depth: Int) -> GPUMatrix3
	{
		precondition(
			width * height * depth == Int(descriptor.width) * Int(descriptor.height) * Int(descriptor.depth),
			"Number of values in matrix must be equal."
		)
		
		return GPUMatrix3(descriptor: (width: UInt32(width), height: UInt32(height), depth: UInt32(depth)), buffer: buffer)
	}
	
	func reshaped(descriptor: (width: UInt32, height: UInt32, depth: UInt32), descriptorBuffer: MTLBuffer) -> GPUMatrix3
	{
		return GPUMatrix3(descriptor: descriptor, buffer: self.buffer, descriptorBuffer: descriptorBuffer)
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

@available(OSX 10.12, *)
extension GPUMatrix3: CustomStringConvertible
{
	public var description: String
	{
		return "GPUMatrix: \(self.descriptor)"
	}
}


@available(OSX 10.12, *)
extension GPUMatrix: CustomStringConvertible
{
	public var description: String
	{
		return "GPUMatrix: \(self.descriptor)"
	}
}


@available(OSX 10.12, *)
public enum GPUTensor
{
	case vector(MTLBuffer, length: Int)
	case matrix(GPUMatrix)
	case matrix3(GPUMatrix3)
	
	internal var count: UInt32
	{
		switch self
		{
		case .vector(_, length: let length):
			return UInt32(length)
			
		case .matrix(let matrix):
			let descriptor = matrix.descriptor
			return descriptor.width * descriptor.height
			
		case .matrix3(let matrix):
			let descriptor = matrix.descriptor
			return descriptor.width * descriptor.height * descriptor.depth
		}
	}
	
	internal var buffer: MTLBuffer
	{
		switch self
		{
		case .vector(let buffer, length: _):
			return buffer
			
		case .matrix(let matrix):
			return matrix.buffer
			
		case .matrix3(let matrix):
			return matrix.buffer
		}
	}
}

@available(OSX 10.12, *)
extension GPUTensor: CustomStringConvertible
{
	public var description: String
	{
		switch self
		{
		case .vector(_, length: let count):
			return "GPUTensor (vector: \(count) elements)"
			
		case .matrix(let matrix):
			return "GPUTensor (\(matrix))"
			
			
		case .matrix3(let matrix):
			return "GPUTensor (\(matrix))"
		}
	}
}
