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


/// Retrieves an element from a matrix at the given column and row.
///
/// This method assumes that the matrix values are stored in row major data ordering
///
/// - parameter matrix_descriptor: Descriptor containing the dimensions of the matrix
/// - parameter matrix_store: Value store of the matrix.
/// - parameter column: Column at which the matrix should be accessed
/// - parameter row: Row at which the matrix should be accessed
/// - returns: Value stored in the matrix at the given column and row
float matrix_get(matrix_t matrix_descriptor, const device float* matrix_store, uint column, uint row)
{
	return matrix_store[(int)(matrix_descriptor.width * row + column)];
}


/// Sets an element in a matrix at the given column and row.
///
/// This method assumes that the matrix values are stored in row major data ordering
///
/// - parameter matrix_descriptor: Descriptor containing the dimensions of the matrix
/// - parameter matrix_store: Value store of the matrix.
/// - parameter column: Column at which the matrix should be set
/// - parameter row: Row at which the matrix should be set
/// - parameter value: New value which will replace the value at the specified location in the matrix.
void matrix_set(matrix_t matrix_descriptor, device float* matrix_store, uint column, uint row, float value)
{
	matrix_store[(int)(matrix_descriptor.width * row + column)] = value;
}


/// Retrieves an element from a three dimensional matrix at the given column and row.
///
/// This method assumes that the matrix values are stored in slice major and then in row major data ordering
///
/// - parameter matrix_descriptor: Descriptor containing the dimensions of the matrix
/// - parameter matrix_store: Value store of the matrix.
/// - parameter column: Column at which the matrix should be accessed
/// - parameter row: Row at which the matrix should be accessed
/// - parameter slice: Slice at which the matrix should be accessed
/// - returns: Value stored in the matrix at the given column and row
float matrix3_get(matrix3_t matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice)
{
	return matrix_store[(int)((matrix_descriptor.height * slice + row) * matrix_descriptor.width + column)];
}


/// Sets an element in a three dimensional matrix at the given column and row.
///
/// This method assumes that the matrix values are stored in slice major and then in row major data ordering
///
/// - parameter matrix_descriptor: Descriptor containing the dimensions of the matrix
/// - parameter matrix_store: Value store of the matrix.
/// - parameter column: Column at which the matrix should be set
/// - parameter row: Row at which the matrix should be set
/// - parameter slice: Slice at which the matrix should beset
/// - parameter value: New value which will replace the value at the specified location in the matrix.
void matrix3_set(matrix3_t matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value)
{
	matrix_store[(int)((matrix_descriptor.height * slice + row) * matrix_descriptor.width + column)] = value;
}

