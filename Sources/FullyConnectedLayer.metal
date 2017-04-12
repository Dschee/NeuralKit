//
//  FullyConnectedLayer.metal
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

//MARK: Layer forward propagation functions

/**
 
 Forwards inputs through a fully connected layer.
 
 This kernel needs to be called for each output value which should be generated.
 
 - parameter input: Values present at input layer. Must be a matrix with the dimensions (1, 1, inputCount)
 - parameter inputDescriptor: Matrix descriptor for input. Must have a width and height of 1.
 - parameter output: Outputs which will be calculated by this feed forward layer. Must have the size (1, 1, outputCount).
 - parameter outputDescriptor: Matrix descriptor for output. Must have a width and height of 1.
 - parameter weights: Weight matrix of the feed forward layer.
 The width must be equal to the input depth + 1 for bias values.
 The height must be equal to the number of outputs of this layer.
 - parameter weightDescriptor: Descriptor for the weight matrix.
 The width must be equal to the input depth + 1 for bias values.
 The height must be equal to the number of outputs of this layer.
 - parameter row: Global ID of the kernel invokation. Indicates the index of the output which will be updated by the current invokation.
 
 */
kernel void FullyConnectedLayer_forward(const	device	float*		input				[[buffer(0)]],
										constant		matrix3_t	&input_descriptor	[[buffer(1)]],
												device	float*		output				[[buffer(2)]],
										constant		matrix3_t	&output_descriptor	[[buffer(3)]],
										const	device	float*		weights				[[buffer(4)]],
										constant		matrix_t	&weight_descriptor	[[buffer(5)]],
														uint		row					[[thread_position_in_grid]])
{
	if (row >= output_descriptor.depth)
		return;
	
	// Calculating single index of matrix vector product
	float sum = 0;
	
	for (uint column = 0; column < input_descriptor.depth; column++)
	{
		sum += matrix3_get(input_descriptor, input, 0, 0, column) * matrix_get(weight_descriptor, weights, column, row);
	}
	
	// Adding bias
	sum += matrix_get(weight_descriptor, weights, input_descriptor.depth, row);
	
	matrix3_set(output_descriptor, output, 0, 0, row, sum);
}


kernel void FullyConnectedLayer_backpropagate(const	device	float*		input						[[buffer(0)]],
											  constant		matrix3_t	&input_descriptor			[[buffer(1)]],
											  const	device	float*		next_gradient				[[buffer(2)]],
											  constant		matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
													device	float*		gradient					[[buffer(4)]], // skip gradient descriptor
													device	float*		weights						[[buffer(6)]],
											  constant		matrix_t	&weight_descriptor			[[buffer(7)]],
											 	 			uint		column						[[thread_position_in_grid]])
{
	if (column >= input_descriptor.depth)
		return;
	
	float grad = 0;
	
	for (uint row = 0; row < next_gradient_descriptor.depth; row++)
	{
		grad += matrix3_get(next_gradient_descriptor, next_gradient, 0, 0, row) * matrix_get(weight_descriptor, weights, column, row);
	}
	
	matrix3_set(input_descriptor, gradient, 0, 0, column, grad);
}

kernel void FullyConnectedLayer_update_gradients(const	device	float*		input						[[buffer(0)]],
												 constant		matrix3_t	&input_descriptor			[[buffer(1)]],
												 const	device	float*		next_gradient				[[buffer(2)]],
												 constant		matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
												 		device	float*		weight_gradient				[[buffer(8)]],
												 constant		matrix_t	&weight_descriptor			[[buffer(9)]],
											   					uint2		pos							[[thread_position_in_grid]])
{
	uint column = pos[0];
	uint row = pos[1];
	
	if (column > input_descriptor.depth || row >= next_gradient_descriptor.depth)
		return;
	
	float in = (column == input_descriptor.depth) ? 1 : matrix3_get(input_descriptor, input, 0, 0, column);
	
	float weightGradient = matrix_get(weight_descriptor, weight_gradient, column, row);
	weightGradient += matrix3_get(next_gradient_descriptor, next_gradient, 0, 0, row) * in;
	matrix_set(weight_descriptor, weight_gradient, column, row, weightGradient);
}
