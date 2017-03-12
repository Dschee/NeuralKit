//
//  Matrix3.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 10.03.17.
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
import Accelerate

/// A 3 dimensional matrix (tensor)
public struct Matrix3
{
	
	/// Values stored in the matrix.
	/// The element at position column: n, row: m, slice: l will be at n + width * m + height * width * l
	public internal(set) var values: [Float]
	
	
	/// Width of the tensor
	public let width: Int
	
	
	/// Height of the tensor
	public let height: Int
	
	
	/// Depth of the tensor
	public let depth: Int
	
	
	/// Dimension of the tensor
	public var dimension:(width: Int, height: Int, depth: Int)
	{
		return (width: width, height: height, depth: depth)
	}
	
	
	/// Indices of the tensor in the form (column,row,slice)
	///
	/// This can be used to avoid nested loops on index based iteration:
	///
	///		for (column, row, slice) in matrix.indices {
	///			let value = matrix[column, row, slice]
	///			...
	///		}
	///
	/// The number of possible indices is equal to width * height * depth of the matrix
	public var indices:[(Int,Int,Int)]
	{
		func combine<BoundA: Comparable, BoundB: Comparable, BoundC: Comparable>(_ a: CountableRange<BoundA>, _ b: CountableRange<BoundB>, _ c: CountableRange<BoundC>) -> [(BoundA,BoundB,BoundC)]
		{
			return c.flatMap{elC in b.flatMap{elB in a.map{($0,elB,elC)}}}
		}
		return combine(0..<width,0..<height,0..<depth)
	}
	
	
	/// Initializes a three dimensional matrix from the given values and dimensions
	///
	/// The number of values must be equal to width * height * depth
	///
	/// The element at position column: n, row: m, slice: l must be at n + width * m + height * width * l
	/// in the provided value vector.
	///
	/// - Parameters:
	///   - values: Values which will be stored in the matrix
	///   - width: Width of the matrix
	///   - height: Height of the matrix
	///   - depth: Depth of the matrix
	public init(values: [Float], width: Int, height: Int, depth: Int)
	{
		precondition(width * height * depth == values.count, "Matrix dimensions are incorrect")
		
		self.values = values
		self.width = width
		self.height = height
		self.depth = depth
	}
	
	
	/// Initializes a three dimensional matrix from a vector of slices.
	///
	/// The slices must be vectors containing vectors of columns of the matrix
	///
	/// - Parameter values: Slices from which the matrix will be initialized
	public init(values: [[[Float]]])
	{
		self.values = values.flatMap{$0.flatMap{$0}}
		self.width = values.first?.first?.count ?? 0
		self.height = values.first?.count ?? 0
		self.depth = values.count
		assert(self.values.count == self.width * self.height * self.depth, "Dimension of matrix does not match number of elements provided.")
	}
	
	
	/// Initializes a three dimensional matrix and sets every value to the repeating value.
	///
	/// - Parameters:
	///   - value: Value to which every element of the matrix will be set
	///   - width: Width of the matrix
	///   - height: Height of the matrix
	///   - depth: Depth of the matrix
	public init(repeating value: Float, width: Int, height: Int, depth: Int)
	{
		self.values = [Float](repeating: value, count: width * height * depth)
		self.width = width
		self.height = height
		self.depth = depth
	}
	
	
	/// Initializes a three dimensional matrix from a two dimensional matrix
	/// with equal dimensions.
	///
	/// - Parameter matrix: Source matrix
	public init(_ matrix: Matrix)
	{
		self.width = matrix.width
		self.height = matrix.height
		self.depth = 1
		self.values = matrix.values
	}
	
	
	/// Subscript to retrieve or set a single element of the matrix
	///
	/// - Parameters:
	///   - x: Column of the element
	///   - y: Row of the element
	///   - z: Slice of the element
	public subscript(x: Int, y: Int, z: Int) -> Float
	{
		get
		{
			return values[width * (height * z + y) + x]
		}
		
		set (new)
		{
			values[width * (height * z + y) + x] = new
		}
	}
	
	
	/// Subscript to retrieve or set a three dimensional submatrix of the matrix
	///
	/// If the position of elements from the submatrix exceeds the bounds of the matrix,
	/// the element will be set to zero on retrieval or ignored when copied into the matrix.
	///
	/// - Parameters:
	///   - column: Column at which the submatrix starts
	///   - row: Row at which the submatrix starts
	///   - slice: Slice at which the submatrix starts
	///   - width: Width of the submatrix
	///   - height: Height of the submatrix
	///   - depth: Depth of the submatrix
	public subscript(x column: Int, y row: Int, z slice: Int, width width: Int, height height: Int, depth depth: Int) -> Matrix3
	{
		get
		{
			var result = Matrix3(repeating: 0, width: width, height: height, depth: depth)
			for (x,y,z) in result.indices
				where 0 ..< self.depth ~= z + slice && 0 ..< self.height ~= y + row && 0 ..< self.width ~= x + column
			{
				result[x,y,z] = self[x+column, y+row, z+slice]
			}
			return result
		}
		
		set (new)
		{
			for (x,y,z) in new.indices
				where 0 ..< self.depth ~= z + slice && 0 ..< self.height ~= y + row && 0 ..< self.width ~= x + column
			{
				values[self.width * (self.height * (slice + z) + row + y) + column + x] = new.values[width * (height * z + y) + x]
			}
		}
	}
	
	
	/// Reverses the matrix. The element at position (i,j,k) will be at (width-i,height-j,depth-k)
	/// in the resulting matrix.
	///
	/// - Returns: Matrix generated by reversing a matrix.
	public func reversed() -> Matrix3
	{
		return Matrix3(values: self.values.reversed(), width: self.width, height: self.height, depth: self.depth)
	}
	
	
	/// Performs a component wise addition of two matrices
	///
	/// - Parameters:
	///   - lhs: First summand
	///   - rhs: Second summand
	/// - Returns: Sum of the input matrices generated by adding elements at equal indices.
	public static func + (lhs: Matrix3, rhs: Matrix3) -> Matrix3
	{
		precondition(lhs.dimension == rhs.dimension, "Dimensions of source matrices must be equal.")
		
		return Matrix3(values: lhs.values &+ rhs.values, width: lhs.width, height: lhs.height, depth: lhs.depth)
	}
	
	
	public static func += (lhs: inout Matrix3, rhs: Matrix3)
	{
		precondition(lhs.dimension == rhs.dimension, "Dimensions of source matrices must be equal.")
		
		lhs.values = lhs.values &* rhs.values
	}
	
	
	/// Returns a matrix generated from a matrix by applying the transform function to every element
	///
	/// - Parameter transform: Transform function which will be applied on every element
	/// - Returns: Matrix generated by applying the transform function to the elements of the initial matrix
	public func map(_ transform: (Float) throws -> Float) rethrows -> Matrix3
	{
		return try Matrix3(values: self.values.map(transform), width: self.width, height: self.height, depth: self.depth)
	}
	
	
	/// Returns a matrix generated from a matrix by applying the vectorized transform function to every element
	///
	/// - Parameter transform: Vectorized transform function which will be applied on every element
	/// - Returns: Matrix generated by applying the transform function to the elements of the initial matrix
	public func mapv(_ transform: ([Float]) throws -> [Float]) rethrows -> Matrix3
	{
		return try Matrix3(values: transform(self.values), width: width, height: height, depth: depth)
	}
	
	
	/// Reshapes the matrix into a new matrix containing the same values.
	///
	/// The reshaped matrix must store the same number of values as the source matrix
	///
	/// - Parameters:
	///   - width: Width of the reshaped matrix
	///   - height: Height of the reshaped matrix
	///   - depth: Depth of the reshaped matrix
	/// - Returns: Reshaped matrix
	public func reshaped(width: Int, height: Int, depth: Int) -> Matrix3
	{
		precondition(
			width * height * depth == self.width * self.height * self.depth,
			"Number of values in reshaped matrix must be equal to number of values in source matrix"
		)
		
		return Matrix3(values: self.values, width: width, height: height, depth: depth)
	}
	
	
	/// Performs a convolution of the matrix using the provided convolution kernel
	///
	/// - Parameters:
	///   - kernel: Convolution kernel
	///   - horizontalStride: Horizontal stride at which the source matrix is traversed
	///   - verticalStride: Vertical stride at which the source matrix is traversed
	///   - lateralStride: Lateral stride at which the source matrix is traversed
	///   - horizontalInset: Horizontal inset at which the traversion begins and ends
	///   - verticalInset: Vertical inset at which the traversion begins and ends
	///   - lateralInset: Lateral inset at which the traversion begins and ends
	/// - Returns: Result of the convolution operation
	public func convolved(
		with kernel: Matrix3,
		horizontalStride: Int = 1,
		verticalStride: Int = 1,
		lateralStride: Int = 1,
		horizontalInset: Int = 0,
		verticalInset: Int = 0,
		lateralInset: Int = 0
	) -> Matrix3
	{
		var output = Matrix3(
			repeating: 0,
			width:  self.width  / horizontalStride - kernel.width  + 1 - 2 * horizontalInset,
			height: self.height / verticalStride   - kernel.height + 1 - 2 * verticalInset,
			depth:  self.depth  / lateralStride    - kernel.depth  + 1 - 2 * lateralInset
		)
		
		for (x,y,z) in output.indices
		{
			let source = self[
				x:		x * horizontalStride + horizontalInset,
				y:		y * verticalStride	 + verticalInset,
				z:		z * lateralStride	 + lateralInset,
				width:	kernel.width,
				height: kernel.height,
				depth:	kernel.depth
			]
			output[x,y,z] = source.values * kernel.values
		}
		
		return output
	}
	
	
	/// Performs a correlation
	///
	/// - Parameters:
	///   - kernel: Kernel with which the matrix is correlated
	///   - horizontalStride: Horizontal stride at which the destination matrix is traversed
	///   - verticalStride: Vertical stride at which the destination matrix is traversed
	///   - lateralStride: Lateral stride at which the destination matrix is traversed
	///   - horizontalInset: Horizontal inset at which the traversion begins and ends
	///   - verticalInset: Vertical inset at which the traversion begins and ends
	///   - lateralInset: Lateral inset at which the traversion begins and ends
	/// - Returns: Result of the correlation operation
	public func correlated(
		with kernel: Matrix3,
		horizontalStride: Int = 1,
		verticalStride: Int = 1,
		lateralStride: Int = 1,
		horizontalInset: Int = 0,
		verticalInset: Int = 0,
		lateralInset: Int = 0
	) -> Matrix3
	{
		var result = Matrix3(
			repeating: 0,
			width:  self.width  * horizontalStride + kernel.width  - 1 + 2 * horizontalInset,
			height: self.height * verticalStride   + kernel.height - 1 + 2 * verticalInset,
			depth:  self.depth  * lateralStride    + kernel.depth  - 1 + 2 * lateralInset
		)
		
		let reversedKernel = kernel.reversed()
		
		for (x,y,z) in self.indices
		{
			let source = self[x,y,z]
			let correlated = reversedKernel.mapv{$0 &* source}
			result[
				x: x * horizontalStride + horizontalInset,
				y: y * verticalStride + verticalInset,
				z: z * lateralStride + lateralInset,
				width: kernel.width,
				height: kernel.height,
				depth: kernel.depth
			] += correlated
		}
		
		return result
	}
	
}


extension Matrix3: CustomStringConvertible
{
	
	/// A textual representation of this instance.
	///
	/// Instead of accessing this property directly, convert an instance of any
	/// type to a string by using the `String(describing:)` initializer. For
	/// example:
	///
	///     struct Point: CustomStringConvertible {
	///         let x: Int, y: Int
	///
	///         var description: String {
	///             return "(\(x), \(y))"
	///         }
	///     }
	///
	///     let p = Point(x: 21, y: 30)
	///     let s = String(describing: p)
	///     print(s)
	///     // Prints "(21, 30)"
	///
	/// The conversion of `p` to a string in the assignment to `s` uses the
	/// `Point` type's `description` property.
	public var description: String
	{
		return (0 ..< depth).map
			{
				zIndex in
				return (0 ..< height).map
					{
						rowIndex in
						return (0 ..< width)
							.map{self[$0, rowIndex, zIndex]}
							.map{"\($0)"}
							.joined(separator: "\t")
					}
					.joined(separator: "\n")
			}
			.joined(separator: "\n\n")
	}
	
}


extension Matrix3: Equatable
{
	
	/// Returns a Boolean value indicating whether two values are equal.
	///
	/// Equality is the inverse of inequality. For any values `a` and `b`,
	/// `a == b` implies that `a != b` is `false`.
	///
	/// - Parameters:
	///   - lhs: A value to compare.
	///   - rhs: Another value to compare.
	public static func ==(lhs: Matrix3, rhs: Matrix3) -> Bool
	{
		return (lhs.width == rhs.width) &&
			(lhs.height == rhs.height) &&
			(lhs.depth == rhs.depth) &&
			zip(lhs.values, rhs.values).map(==).reduce(true, {$0 && $1})
	}
	
}
