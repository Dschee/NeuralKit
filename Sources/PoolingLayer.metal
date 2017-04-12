//
//  PoolingLayer.metal
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


kernel void PoolingLayer_forward(const	device	float*		input				[[buffer(0)]],
								 constant		matrix3_t	&input_descriptor	[[buffer(1)]],
								 		device	float*		output				[[buffer(2)]],
								 constant		matrix3_t	&output_descriptor	[[buffer(3)]],
								 				uint3		pos					[[thread_position_in_grid]])
{
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	uint horizontalScale = input_descriptor.width / output_descriptor.width;
	uint verticalScale = input_descriptor.height / output_descriptor.height;
	uint lateralScale = input_descriptor.depth / output_descriptor.depth;
	
	uint sourceX = horizontalScale * pos[0];
	uint sourceY = verticalScale * pos[1];
	uint sourceZ = lateralScale * pos[2];
	
	float max = matrix3_get(input_descriptor, input, sourceX, sourceY, sourceZ);
	
	for (uint z = sourceZ; z < sourceZ + lateralScale; z++)
	{
		for (uint y = sourceY; y < sourceY + lateralScale; y++)
		{
			for (uint x = sourceX; x < sourceX + lateralScale; x++)
			{
				max = fmax(max, matrix3_get(input_descriptor, input, x, y, z));
			}
		}
	}
	
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], max);
}


kernel void PoolingLayer_backpropagate(const device float*		input						[[buffer(0)]],
									   constant		matrix3_t	&input_descriptor			[[buffer(1)]],
									   const device	float*		next_gradient				[[buffer(2)]],
									   constant		matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
											 device	float*		gradient					[[buffer(4)]], // skipping gradient descriptor
									  				uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= next_gradient_descriptor.width || pos[1] >= next_gradient_descriptor.height || pos[2] >= next_gradient_descriptor.depth)
		return;
	
	uint horizontalScale = input_descriptor.width / next_gradient_descriptor.width;
	uint verticalScale = input_descriptor.height / next_gradient_descriptor.height;
	uint lateralScale = input_descriptor.depth / next_gradient_descriptor.depth;
	
	uint sourceX = horizontalScale * pos[0];
	uint sourceY = verticalScale * pos[1];
	uint sourceZ = lateralScale * pos[2];
	
	uint maxX = 0;
	uint maxY = 0;
	uint maxZ = 0;
	
	float max = matrix3_get(input_descriptor, input, sourceX, sourceY, sourceZ);
	
	for (uint z = sourceZ; z < lateralScale + sourceZ; z++)
	{
		for (uint y = sourceY; y < verticalScale + sourceY; y++)
		{
			for (uint x = sourceX; x < horizontalScale + sourceX; x++)
			{
				float in = matrix3_get(input_descriptor, input, x, y, z);
				if (in > max)
				{
					max = in;
					maxX = x;
					maxY = y;
					maxZ = z;
				}
				matrix3_set(input_descriptor, gradient, x, y, z, 0);
			}
		}
	}
	
	float grad = matrix3_get(next_gradient_descriptor, next_gradient, pos[0], pos[1], pos[2]);
	matrix3_set(input_descriptor, gradient, maxX, maxY, maxZ, grad);
}
