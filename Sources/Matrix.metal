//
//  Matrix.metal
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

#include <metal_stdlib>
using namespace metal;

#include "Matrix.h"

/**
 
 Retrieves an element of a 2D matrix.
 
 This method assumes that the matrix values are stored in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - returns: The value stored at (column, row) in the matrix
 
 */
float matrix_get(matrix_t matrix_descriptor, const device float* matrix_store, uint column, uint row)
{
	return matrix_store[(int)(matrix_descriptor.width * row + column)];
}

/**
 
 Sets an element of a 2D matrix.
 
 This method assumes that the matrix values are stored in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - parameter value: The value to which the matrix at (column, row) should be set to
 
 */
void matrix_set(matrix_t matrix_descriptor, device float* matrix_store, uint column, uint row, float value)
{
	matrix_store[(int)(matrix_descriptor.width * row + column)] = value;
}


/**
 
 Retrieves an element of a 3D matrix.
 
 This method assumes that the matrix values are stored in slice major and then in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - parameter slice: The z index at which the matrix should be accessed
 - returns: The value stored at (column, row) in the matrix
 
 */
float matrix3_get(matrix3_t matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice)
{
	return matrix_store[(int)((matrix_descriptor.height * slice + row) * matrix_descriptor.width + column)];
}


/**
 
 Sets an element of a 3D matrix.
 
 This method assumes that the matrix values are stored in slice major and then in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - parameter slice: The z index at which the matrix should be accessed
 - parameter value: The value to which the matrix at (column, row) should be set to
 
 */
void matrix3_set(matrix3_t matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value)
{
	matrix_store[(int)((matrix_descriptor.height * slice + row) * matrix_descriptor.width + column)] = value;
}

