//
//  NonlinearityLayer.metal
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
 
 Forwards inputs through a nonlinearity layer using the sigmoid function.
 
 This kernel needs to be called for each input / output value of this layer.
 
 - parameter input: Values present at input layer.
 - parameter inputDescriptor: Matrix descriptor for input.
 - parameter output: Outputs which will be calculated by this feed forward layer.
 - parameter outputDescriptor: Matrix descriptor for output.
 - parameter pos: Global ID of the kernel invokation. Indicates the index of the output which will be updated by the current invokation.
 
 */
kernel void NonlinearityLayer_forward_sigmoid(const	device	float*		input				[[buffer(0)]],
											  constant		matrix3_t	&input_descriptor	[[buffer(1)]],
											  		device	float*		output				[[buffer(2)]],
											  constant		matrix3_t	&output_descriptor	[[buffer(3)]],
											  				uint3		pos					[[thread_position_in_grid]])
{
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	float result = 1 / (1 - exp(-matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2])));
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], result);
}


kernel void NonlinearityLayer_backpropagate_sigmoid(const	device	float*		next_gradient				[[buffer(0)]],
													constant		matrix3_t	&next_gradient_descriptor	[[buffer(1)]],
													const	device	float*		outputs						[[buffer(2)]],
															device	float*		gradient					[[buffer(4)]], // Skip gradient descriptor
																	uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= next_gradient_descriptor.width || pos[1] >= next_gradient_descriptor.height || pos[2] >= next_gradient_descriptor.depth)
		return;
	
	float grad = matrix3_get(next_gradient_descriptor, next_gradient, pos[0], pos[1], pos[2]);
	float out = matrix3_get(next_gradient_descriptor, outputs, pos[0], pos[1], pos[2]);
	grad *= out * (1 - out);
	matrix3_set(next_gradient_descriptor, gradient, pos[0], pos[1], pos[2], grad);
}


/**
 
 Forwards inputs through a nonlinearity layer using the tanh function.
 
 This kernel needs to be called for each input / output value of this layer.
 
 - parameter input: Values present at input layer.
 - parameter inputDescriptor: Matrix descriptor for input.
 - parameter output: Outputs which will be calculated by this feed forward layer.
 - parameter outputDescriptor: Matrix descriptor for output.
 - parameter pos: Global ID of the kernel invokation. Indicates the index of the output which will be updated by the current invokation.
 
 */
kernel void NonlinearityLayer_forward_tanh(const	device	float*		input				[[buffer(0)]],
										   constant			matrix3_t	&input_descriptor	[[buffer(1)]],
										   			device	float*		output				[[buffer(2)]],
										   constant			matrix3_t	&output_descriptor	[[buffer(3)]],
										   					uint3		pos					[[thread_position_in_grid]])
{
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	float result = tanh(matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2]));
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], result);
}

kernel void NonlinearityLayer_backpropagate_tanh(const	device	float*		next_gradient				[[buffer(0)]],
												 constant		matrix3_t	&next_gradient_descriptor	[[buffer(1)]],
												const	device	float*		outputs						[[buffer(2)]],
												 		device	float*		gradient					[[buffer(4)]], // Skip gradient descriptor
												 				uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= next_gradient_descriptor.width || pos[1] >= next_gradient_descriptor.height || pos[2] >= next_gradient_descriptor.depth)
		return;
	
	float grad = matrix3_get(next_gradient_descriptor, next_gradient, pos[0], pos[1], pos[2]);
	float out = matrix3_get(next_gradient_descriptor, outputs, pos[0], pos[1], pos[2]);
	grad *= (1 - out * out);
	matrix3_set(next_gradient_descriptor, gradient, pos[0], pos[1], pos[2], grad);
}


/**
 
 Forwards inputs through a nonlinearity layer using the relu function.
 
 This kernel needs to be called for each input / output value of this layer.
 
 - parameter input: Values present at input layer.
 - parameter inputDescriptor: Matrix descriptor for input.
 - parameter output: Outputs which will be calculated by this feed forward layer.
 - parameter outputDescriptor: Matrix descriptor for output.
 - parameter pos: Global ID of the kernel invokation. Indicates the index of the output which will be updated by the current invokation.
 
 */
kernel void NonlinearityLayer_forward_relu(const	device	float*		input				[[buffer(0)]],
										   constant			matrix3_t	&input_descriptor	[[buffer(1)]],
										   			device	float*		output				[[buffer(2)]],
										   constant			matrix3_t	&output_descriptor	[[buffer(3)]],
										  					uint3		pos					[[thread_position_in_grid]])
{
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	float result = fmax(matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2]), 0);
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], result);
}

kernel void NonlinearityLayer_backpropagate_relu(const	device	float*		next_gradient				[[buffer(0)]],
												 constant		matrix3_t	&next_gradient_descriptor	[[buffer(1)]],
												const	device	float*		outputs						[[buffer(2)]],
														device	float*		gradient					[[buffer(4)]], // Skip gradient descriptor
												 				uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= next_gradient_descriptor.width || pos[1] >= next_gradient_descriptor.height || pos[2] >= next_gradient_descriptor.depth)
		return;
	
	float grad = matrix3_get(next_gradient_descriptor, next_gradient, pos[0], pos[1], pos[2]);
	float out = matrix3_get(next_gradient_descriptor, outputs, pos[0], pos[1], pos[2]);
	grad *= fmax(out, 0);
	matrix3_set(next_gradient_descriptor, gradient, pos[0], pos[1], pos[2], grad);
}
