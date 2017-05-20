//
//  Vector.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 22.04.17.
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

public class Vector: ExpressibleByArrayLiteral
{
	internal var values: UnsafeMutablePointer<Float>
	public let count: Int
	
	public init(_ values: [Float])
	{
		self.values = UnsafeMutablePointer.allocate(capacity: values.count)
		self.values.assign(from: values, count: values.count)
		self.count = values.count
	}
	
	public convenience init(repeating value: Float, count: Int)
	{
		self.init(Array<Float>(repeating: value, count: count))
	}
	
	public typealias Element = Float
	
	public required convenience init(arrayLiteral elements: Vector.Element...)
	{
		self.init(elements)
	}
	
	deinit
	{
		values.deallocate(capacity: count)
	}
	
	public static func add(_ lhs: Vector, _ rhs: Vector, result: inout Vector)
	{
		precondition(lhs.count == rhs.count && lhs.count == result.count, "Both input vectors and result vector must be of equal length")
		vDSP_vadd(lhs.values, 1, rhs.values, 1, result.values, 1, UInt(lhs.count))
	}
	
	public static func add(_ lhs: Vector, _ rhs: Float, result: inout Vector)
	{
		precondition(lhs.count == result.count, "Input vector and result vector must be of equal length")
		vDSP_vsadd(lhs.values, 1, [rhs], result.values, 1, UInt(lhs.count))
	}
	
	@inline(__always)
	public static func add(_ lhs: Float, _ rhs: Vector, result: inout Vector)
	{
		Vector.add(rhs, lhs, result: &result)
	}

	public static func subtract(_ lhs: Vector, _ rhs: Vector, result: inout Vector)
	{
		precondition(lhs.count == rhs.count && lhs.count == result.count, "Both input vectors and result vector must be of equal length")
		vDSP_vsub(rhs.values, 1, lhs.values, 1, result.values, 1, UInt(lhs.count))
	}
	
	public static func subtract(_ lhs: Vector, _ rhs: Float, result: inout Vector)
	{
		precondition(lhs.count == result.count, "Input vector and result vector must be of equal length")
		vDSP_vsadd(lhs.values, 1, [rhs], result.values, 1, UInt(lhs.count))
	}
	
	public static func subtract(_ lhs: Float, _ rhs: Vector, result: inout Vector)
	{
		precondition(rhs.count == result.count, "Input vector and result vector must be of equal length")
		rhs.negate(into: &result)
		add(result, lhs, result: &result)
	}
	
	public static func multiply(_ lhs: Vector, _ rhs: Vector, result: inout Vector)
	{
		precondition(lhs.count == rhs.count && lhs.count == result.count, "Both input vectors and result vector must be of equal length")
		vDSP_vmul(lhs.values, 1, rhs.values, 1, result.values, 1, UInt(lhs.count))
	}
	
	public static func multiply(_ lhs: Vector, _ rhs: Float, result: inout Vector)
	{
		precondition(lhs.count == result.count, "Input vector and result vector must be of equal length")
		vDSP_vsmul(lhs.values, 1, [rhs], result.values, 1, UInt(lhs.count))
	}
	
	public static func multiply(_ lhs: Float, _ rhs: Vector, result: inout Vector)
	{
		multiply(lhs, rhs, result: &result)
	}
	
	public static func divide(_ lhs: Vector, _ rhs: Vector, result: inout Vector)
	{
		precondition(lhs.count == rhs.count && lhs.count == result.count, "Both input vectors and result vector must be of equal length")
		vDSP_vdiv(rhs.values, 1, lhs.values, 1, result.values, 1, UInt(lhs.count))
	}
	
	public static func divide(_ lhs: Vector, _ rhs: Float, result: inout Vector)
	{
		precondition(lhs.count == result.count, "Input vector and result vector must be of equal length")
		vDSP_vsdiv(lhs.values, 1, [rhs], result.values, 1, UInt(lhs.count))
	}
	
	public static func divide(_ lhs: Float, _ rhs: Vector, result: inout Vector)
	{
		precondition(rhs.count == result.count, "Input vector and result vector must be of equal length")
		vDSP_svdiv([lhs], rhs.values, 1, result.values, 1, UInt(rhs.count))
	}
	
	public func negate(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		vDSP_vneg(self.values, 1, result.values, 1, UInt(self.count))
	}
	
	public func sqrt(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		vvsqrtf(result.values, self.values, [Int32(self.count)])
	}
	
	public func rsqrt(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		vvrsqrtf(result.values, self.values, [Int32(self.count)])
	}
	
	public func squared(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		vDSP_vsq(self.values, 1, result.values, 1, UInt(self.count))
	}
	
	public func exp(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		vvexpf(result.values, self.values, [Int32(self.count)])
	}
	
	public func log(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		vvlogf(result.values, self.values, [Int32(self.count)])
	}
	
	public func tanh(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		vvtanhf(result.values, self.values, [Int32(self.count)])
	}
	
	public func tanh_derivative(into result: inout Vector)
	{
		precondition(self.count == result.count, "Source and result vector must be of equal length")
		fatalError()
	}
	
	public func sum() -> Float
	{
		var result: Float = 0
		vDSP_sve(self.values, 1, &result, UInt(self.count))
		return result
	}
}

extension Vector: CustomStringConvertible
{
	public var description: String
	{
		var stringRep = "("
		
		for i in 0 ..< count
		{
			stringRep.append("\(values[i])")
			if i + 1 < count
			{
				stringRep.append(", ")
			}
		}
		stringRep.append(")")
		return stringRep
	}
}


infix operator <- : AssignmentPrecedence

public typealias VectorArithmeticContext = (inout Vector) -> ()

public func <- (lhs: inout Vector, rhs: VectorArithmeticContext)
{
	rhs(&lhs)
}

public prefix func - (values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		values.negate(into: &result)
	}
}

public prefix func - (values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result.negate(into: &result)
	}
}

public func + (lhs: Vector, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.add(lhs, rhs, result: &result)
	}
}

public func + (lhs: Vector, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		Vector.add(lhs, rhs, result: &result)
	}
}

public func + (lhs: Float, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.add(lhs, rhs, result: &result)
	}
}

public func + (lhs: Vector, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.add(lhs, result, result: &result)
	}
}

public func + (lhs: @escaping VectorArithmeticContext, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.add(result, rhs, result: &result)
	}
}

public func + (lhs: Float, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.add(lhs, result, result: &result)
	}
}

public func + (lhs: @escaping VectorArithmeticContext, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.add(result, rhs, result: &result)
	}
}

public func - (lhs: Vector, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.subtract(lhs, rhs, result: &result)
	}
}

public func - (lhs: Vector, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		Vector.subtract(lhs, rhs, result: &result)
	}
}

public func - (lhs: Float, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.subtract(lhs, rhs, result: &result)
	}
}

public func - (lhs: Vector, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.subtract(lhs, result, result: &result)
	}
}

public func - (lhs: @escaping VectorArithmeticContext, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.subtract(result, rhs, result: &result)
	}
}

public func - (lhs: Float, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.subtract(lhs, result, result: &result)
	}
}

public func - (lhs: @escaping VectorArithmeticContext, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.subtract(result, rhs, result: &result)
	}
}

public func * (lhs: Vector, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.multiply(lhs, rhs, result: &result)
	}
}

public func * (lhs: Vector, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		Vector.multiply(lhs, rhs, result: &result)
	}
}

public func * (lhs: Float, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.multiply(lhs, rhs, result: &result)
	}
}

public func * (lhs: Vector, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.multiply(lhs, result, result: &result)
	}
}

public func * (lhs: @escaping VectorArithmeticContext, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.multiply(result, rhs, result: &result)
	}
}

public func * (lhs: Float, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.multiply(lhs, result, result: &result)
	}
}

public func * (lhs: @escaping VectorArithmeticContext, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.multiply(result, rhs, result: &result)
	}
}

public func / (lhs: Vector, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.divide(lhs, rhs, result: &result)
	}
}

public func / (lhs: Vector, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		Vector.divide(lhs, rhs, result: &result)
	}
}

public func / (lhs: Float, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		Vector.divide(lhs, rhs, result: &result)
	}
}

public func / (lhs: Vector, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.divide(lhs, result, result: &result)
	}
}

public func / (lhs: @escaping VectorArithmeticContext, rhs: Vector) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.divide(result, rhs, result: &result)
	}
}

public func / (lhs: Float, rhs: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		rhs(&result)
		Vector.divide(lhs, result, result: &result)
	}
}

public func / (lhs: @escaping VectorArithmeticContext, rhs: Float) -> VectorArithmeticContext
{
	return {
		result in
		lhs(&result)
		Vector.divide(result, rhs, result: &result)
	}
}

public func exp(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		values.exp(into: &result)
	}
}

public func exp(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result.exp(into: &result)
	}
}

public func log(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		values.log(into: &result)
	}
}

public func log(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result.log(into: &result)
	}
}

public func sqrt(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		values.sqrt(into: &result)
	}
}

public func sqrt(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result.sqrt(into: &result)
	}
}

public func rsqrt(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		values.rsqrt(into: &result)
	}
}

public func rsqrt(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result.rsqrt(into: &result)
	}
}

public func tanh(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		values.tanh(into: &result)
	}
}

public func tanh(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result.tanh(into: &result)
	}
}

public func tanh_deriv(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		result <- 1 - squared(values)
	}
}

public func tanh_deriv(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result <- 1 - squared(result)
	}
}

public func squared(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		values.squared(into: &result)
	}
}

public func squared(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		values(&result)
		result.squared(into: &result)
	}
}

public func sigmoid(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		result <- 1 / (1 + exp(-values))
	}
}

public func sigmoid(_ values: @escaping VectorArithmeticContext) -> VectorArithmeticContext
{
	return {
		result in
		result <- 1 / (1 + exp(-values))
	}
}

public func sigmoid_deriv(_ values: Vector) -> VectorArithmeticContext
{
	return {
		result in
		result <- values * (1 - values)
	}
}
