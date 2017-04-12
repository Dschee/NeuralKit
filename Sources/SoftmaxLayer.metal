//
//  SoftmaxLayer.metal
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

#include <metal_stdlib>
#include <metal_math>

using namespace metal;

#include "Matrix.h"


kernel void SoftmaxLayer_forward_exp(const	device	float*		input				[[buffer(0)]],
									 constant		matrix3_t	&input_descriptor	[[buffer(1)]],
											device	float*		output				[[buffer(2)]],
									 constant		matrix3_t	&output_descriptor	[[buffer(3)]],
													uint3		pos					[[thread_position_in_grid]])
{
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	float in = matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2]);
	float exponentiated = exp(in);
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], exponentiated);
}

kernel void SoftmaxLayer_forward(const	device	float*		input				[[buffer(0)]],
								 constant		matrix3_t	&input_descriptor	[[buffer(1)]],
								 device			float*		output				[[buffer(2)]],
								 constant		matrix3_t	&output_descriptor	[[buffer(3)]],
												uint3		pos					[[thread_position_in_grid]])
{
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	float sum = 0;
	
	for (uint z = 0; z < input_descriptor.depth; z++)
	{
		for (uint y = 0; y < input_descriptor.height; y++)
		{
			for (uint x = 0; x < input_descriptor.width; x++)
			{
				sum += matrix3_get(input_descriptor, input, x, y, z);
			}
		}
	}
	
	float in = matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2]);
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], in / sum);
}


