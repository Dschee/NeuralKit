//
//  Matrix.h
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

#ifndef Matrix_h
#define Matrix_h

//MARK: Matrix helper functions

/// A descriptor of a 2D matrix
typedef struct
{
	/// The width of the matrix
	uint width;
	
	/// The height of the matrix
	uint height;
} matrix_t;


/// A descriptor of a 3D matrix
typedef struct
{
	/// The width of the matrix
	uint width;
	
	/// The height of the matrix
	uint height;
	
	/// The depth of the matrix
	uint depth;
} matrix3_t;


float matrix_get(matrix_t matrix_descriptor, const device float* matrix_store, uint column, uint row);

void matrix_set(matrix_t matrix_descriptor, device float* matrix_store, uint column, uint row, float value);


float matrix3_get(matrix3_t matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice);

void matrix3_set(matrix3_t matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value);


#endif /* Matrix_h */
