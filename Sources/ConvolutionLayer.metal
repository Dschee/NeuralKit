//
//  ConvolutionLayer.metal
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


kernel void ConvolutionLayer_forward(const	device	float*		input				[[buffer(0)]],
									 constant		matrix3_t	&input_descriptor	[[buffer(1)]],
									 		device	float*		output				[[buffer(2)]],
									 constant		matrix3_t	&output_descriptor	[[buffer(3)]],
									 const	device	float*		kernels				[[buffer(4)]],
									 const	device	matrix3_t	&kernel_descriptor	[[buffer(5)]],
									 const	device	float*		bias_values			[[buffer(6)]],
									 constant		int			&horizontal_inset	[[buffer(7)]],
									 constant		int			&vertical_inset		[[buffer(8)]],
									 constant		int			&horizontal_stride	[[buffer(9)]],
									 constant		int			&vertical_stride	[[buffer(10)]],
									 				uint3		pos					[[thread_position_in_grid]])
{
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	matrix3_t kernel_desc = kernel_descriptor;
	kernel_desc.depth = input_descriptor.depth;
	const device float* kernel_mat = &kernels[(kernel_desc.width + kernel_desc.height + kernel_desc.depth) * pos[2]];
	
	int inputBaseX = pos[0] * horizontal_stride + horizontal_inset;
	int inputBaseY = pos[1] * vertical_stride + vertical_inset;
	
	float convolution_sum = 0;
	
	for (uint y = 0; y < kernel_desc.height; y++)
	{
		if ((int) y + inputBaseY < 0 || (int) y + inputBaseY >= (int) input_descriptor.height)
			continue;
		
		for (uint x = 0; x < kernel_desc.width; x++)
		{
			if ((int) x + inputBaseX < 0 || (int) x + inputBaseX >= (int) input_descriptor.width)
				continue;
			
			for (uint z = 0; z < kernel_desc.depth; z++)
			{
				convolution_sum +=
				matrix3_get(kernel_desc,		kernel_mat, x,				y,				z) *
				matrix3_get(input_descriptor,	input,		x + inputBaseX, y + inputBaseY, z);
			}
		}
	}
	
	// Applying bias
	convolution_sum += bias_values[pos[2]];
	
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], convolution_sum);
}

kernel void ConvolutionLayer_backpropagate(			device	float*		next_gradient				[[buffer(2)]],
										   constant			matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
										   device			float*		gradient					[[buffer(4)]], // Skip gradient descriptor
										   constant			matrix3_t	&input_descriptor			[[buffer(5)]],
										   const	device	float*		kernels						[[buffer(6)]],
										   			device	matrix3_t	&kernel_descriptor			[[buffer(7)]],
										   constant			int			&horizontal_inset			[[buffer(12)]],
										   constant			int			&vertical_inset				[[buffer(13)]],
										   constant			int			&horizontal_stride			[[buffer(14)]],
										   constant			int			&vertical_stride			[[buffer(15)]],
										   					uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= input_descriptor.width || pos[1] >= input_descriptor.height || pos[2] >= input_descriptor.depth)
		return;
	
	matrix3_t kernel_desc = kernel_descriptor;
	kernel_desc.depth = input_descriptor.depth;
	const uint kernel_count = next_gradient_descriptor.depth;
	
	float gradientSum = 0;
	
	for (uint y = 0; y < kernel_desc.height; y++)
	{
		int outputY = pos[1] - vertical_inset - y;
		
		if (outputY < 0 || outputY >= (int) next_gradient_descriptor.height)
			continue;
		
		for (uint x = 0; x < kernel_desc.width; x++)
		{
			int outputX = pos[0] - horizontal_inset - x;
			
			if (outputX < 0 || outputX >= (int) next_gradient_descriptor.width)
				continue;
			
			for (uint z = 0; z < kernel_count; z++)
			{
				float next_grad = matrix3_get(next_gradient_descriptor, next_gradient, outputX, outputY, z);
				float weight = matrix3_get(kernel_descriptor, kernels, x, y, z * kernel_desc.depth);
				
				gradientSum += weight * next_grad;
			}
		}
	}
	
	matrix3_set(input_descriptor, gradient, pos[0], pos[1], pos[2], gradientSum);
}

kernel void ConvolutionLayer_update_gradients(const	device	float*		input						[[buffer(0)]],
											  constant		matrix3_t	&input_descriptor			[[buffer(1)]],
													device	float*		next_gradient				[[buffer(2)]],
											  constant		matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
													device	float*		weight_gradients			[[buffer(8)]],
													device	matrix3_t	&kernel_descriptor			[[buffer(9)]],
											  constant		int			&horizontal_inset			[[buffer(12)]],
											  constant		int			&vertical_inset				[[buffer(13)]],
											  constant		int			&horizontal_stride			[[buffer(14)]],
											  constant		int			&vertical_stride			[[buffer(15)]],
															uint3		pos							[[thread_position_in_grid]]) // Position in convolution kernels
{
	if (pos[0] >= kernel_descriptor.width || pos[1] >= kernel_descriptor.height || pos[2] >= kernel_descriptor.depth)
		return;
	
	matrix3_t kernel_desc = kernel_descriptor;
	kernel_desc.depth = input_descriptor.depth;
	
	const uint outputZ = pos[2] / input_descriptor.depth;
	const uint inputZ = pos[2] % input_descriptor.depth;
	
	float weightGradient = 0;
	
	for (uint y = 0; y < next_gradient_descriptor.height; y++)
	{
		int vertical_input_position = y * vertical_stride + vertical_inset + pos[1];
		
		if (vertical_input_position < 0 || vertical_input_position >= (int) input_descriptor.height)
			continue;
		
		for (uint x = 0; x < next_gradient_descriptor.width; x++)
		{
			int horizontal_input_position = x * horizontal_stride + horizontal_inset + pos[0];
			
			if (horizontal_input_position < 0 || horizontal_input_position >= (int) input_descriptor.width)
				continue;
			
			float next_grad = matrix3_get(next_gradient_descriptor, next_gradient, x, y, outputZ);
			float in_scale = matrix3_get(input_descriptor, input, horizontal_input_position, vertical_input_position, inputZ);
			
			weightGradient += next_grad * in_scale;
		}
	}
	
	float currentGradient = matrix3_get(input_descriptor, weight_gradients, pos[0], pos[1], pos[2]);
	matrix3_set(input_descriptor, weight_gradients, pos[0], pos[1], pos[2], currentGradient + weightGradient);
}

kernel void ConvolutionLayer_update_bias_gradients(			device	float*		next_gradient				[[buffer(2)]],
												   constant			matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
												   			device	float*		weight_gradients			[[buffer(8)]], // Skip kernel deltas descriptor
												   constant			int			&horizontal_inset			[[buffer(12)]],
												   constant			int			&vertical_inset				[[buffer(13)]],
												   constant			int			&horizontal_stride			[[buffer(14)]],
												   constant			int			&vertical_stride			[[buffer(15)]],
																	uint		index						[[thread_position_in_grid]])
{
	if (index >= next_gradient_descriptor.depth)
		return;
	
	float biasGradient = weight_gradients[index];
	
	for (uint y = 0; y < next_gradient_descriptor.height; y++)
	{
		for (uint x = 0; x < next_gradient_descriptor.width; x++)
		{
			biasGradient += matrix3_get(next_gradient_descriptor, next_gradient, x, y, index);
		}
	}
	
	weight_gradients[index] = biasGradient;
}

