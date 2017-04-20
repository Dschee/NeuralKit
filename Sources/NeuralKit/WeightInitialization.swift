//
//  WeightInitialization.swift
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
import MatrixVector


/// Creates a three dimensional weight matrix
/// and fills it with small random values in the specified range
///
/// - Parameters:
///   - width: Width of the weight matrix
///   - height: Height of the weight matrix
///   - depth: Depth of the weight matrix
///   - range: Range in which the random values should be. Default: [-0.01; 0.01]
/// - Returns: Weight matrix containing random values
public func RandomWeightMatrix(width: Int, height: Int, depth: Int, range: ClosedRange<Float> = Float(-0.01) ... Float(0.01)) -> Matrix3
{
	var weightMatrix = Matrix3(repeating: 0, width: width, height: height, depth: depth)
	
	for (x,y,z) in weightMatrix.indices
	{
		weightMatrix[x,y,z] = Float(drand48()) * (range.upperBound - range.lowerBound) + range.lowerBound
	}
	
	return weightMatrix
}

public func RandomWeightMatrix(width: Int, height: Int, depth: Int, variance: Float) -> Matrix3
{
	var weightMatrix = Matrix3(repeating: 0, width: width, height: height, depth: depth)
	
	for (x,y,z) in weightMatrix.indices
	{
		let uniformA = Float(drand48())
		let uniformB = Float(drand48())
		
		let gaussianDistributedRandomNumber = sqrt(-2 * log(uniformA)) * cos(2 * Float.pi * uniformB) * variance
		weightMatrix[x,y,z] = gaussianDistributedRandomNumber
	}
	
	return weightMatrix
}

public func RandomWeightMatrix(width: Int, height: Int, variance: Float) -> Matrix
{
	var weightMatrix = Matrix(repeating: 0, width: width, height: height)
	
	for (x,y) in weightMatrix.indices
	{
		let uniformA = Float(drand48())
		let uniformB = Float(drand48())
		
		let gaussianDistributedRandomNumber = sqrt(-2 * log(uniformA)) * cos(2 * Float.pi * uniformB) * variance
		weightMatrix[x,y] = gaussianDistributedRandomNumber
	}
	
	return weightMatrix
}


/// Creates a weight matrix
/// and fills it with small random values in the specified range
///
/// - Parameters:
///   - width: Width of the weight matrix
///   - height: Height of the weight matrix
///   - range: Range in which the random values should be. Default: [-0.01; 0.01]
/// - Returns: Weight matrix containing random values
public func RandomWeightMatrix(width: Int, height: Int, range: ClosedRange<Float> = Float(-0.01) ... Float(0.01)) -> Matrix
{
	var weightMatrix = Matrix(repeating: 0, width: width, height: height)
	
	for (x,y) in weightMatrix.indices
	{
		weightMatrix[x,y] = Float(drand48()) * (range.upperBound - range.lowerBound) + range.lowerBound
	}
	
	return weightMatrix
}


/// Creates a pertubation matrix
/// consisting of very small values around zero with a few randomly larger values
///
/// - Parameters:
///   - width: Width of the perturbation matrix
///   - height: Height of the perturbation matrix
/// - Returns: Perturbation matrix
public func RandomPertubationMatrix(width: Int, height: Int) -> Matrix
{
	let matA = RandomWeightMatrix(width: width, height: height, range: 0 ... 1)
	let matB = RandomWeightMatrix(width: width, height: height, range: 0 ... 1)
	
	// generating a normal distribution from uniformly distributed random values
	let transformed = matA.mapv{sqrt(-2 &* log($0))}
	let randomNegated = zip(transformed.values, matB.values).map{$1 < 0.5 ? -$0 : $0}
	return Matrix(values: randomNegated, width: width, height: height).mapv{pow($0, Array<Float>(repeating: 5, count: randomNegated.count)) &* 0.0003}
}

