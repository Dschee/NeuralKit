//
//  Optimizers.metal
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


kernel void Optimize_sgd(		device	float*		weights			[[buffer(0)]],
						 constant		uint		&count			[[buffer(1)]],
								device	float*		gradients		[[buffer(2)]],
						 constant		float		&learningRate	[[buffer(3)]],
										uint		index			[[thread_position_in_grid]])
{
	if (index >= count)
		return;
	
	weights[index] -= gradients[index] * learningRate;
	gradients[index] = 0;
}

kernel void Optimize_momentum(		device	float*		weights			[[buffer(0)]],
							  constant		uint		&count			[[buffer(1)]],
							  		device 	float*		gradients		[[buffer(2)]],
							  		device 	float*		weight_deltas	[[buffer(3)]],
							  constant 		float		&learningRate	[[buffer(4)]],
							  constant		float		&momentum		[[buffer(5)]],
							  				uint		index			[[thread_position_in_grid]])
{
	if (index >= count)
		return;
	
	float delta = weight_deltas[index] * momentum + gradients[index] * learningRate;
	weight_deltas[index] = delta;
	weights[index] -= delta;
	gradients[index] = 0;
}

kernel void Optimize_adagrad(		device	float*		weights					[[buffer(0)]],
							 constant		uint		&count					[[buffer(1)]],
									device	float*		gradients				[[buffer(2)]],
									device	float*		squared_gradient_sums	[[buffer(3)]],
							 constant		float		&learningRate			[[buffer(4)]],
											uint		index					[[thread_position_in_grid]])
{
	if (index >= count)
		return;
	
	float gradient = gradients[index];
	
	float squared_gradient_sum = squared_gradient_sums[index];
	squared_gradient_sum += gradient * gradient;
	squared_gradient_sums[index] = squared_gradient_sum;
	
	weights[index] -= learningRate * rsqrt(squared_gradient_sum + 1E-8) * gradient;
	gradients[index] = 0;
}

kernel void Optimize_adadelta(		device	float*		weights						[[buffer(0)]],
							  constant		uint		&count						[[buffer(1)]],
							  		device	float*		gradients					[[buffer(2)]],
							  		device	float*		squared_gradient_sums		[[buffer(3)]],
							  		device	float*		squared_weight_update_sums	[[buffer(4)]],
							  constant		float		&decay						[[buffer(5)]],
							  				uint		index						[[thread_position_in_grid]])
{
	if (index >= count)
		return;
	
	float gradient = gradients[index];
	
	float squared_gradient_sum = (decay * squared_gradient_sums[index]) + ((1 - decay) * (gradient * gradient));
	float squared_weight_update_sum = squared_weight_update_sums[index];
	
	float weight_delta = sqrt(squared_weight_update_sum + 1E-8) * rsqrt(squared_gradient_sum + 1E-8) * gradient;
	squared_weight_update_sums[index] = (squared_weight_update_sum * decay) + ((1 - decay) * (weight_delta * weight_delta));
	
	weights[index] -= weight_delta;
	gradients[index] = 0;
}
