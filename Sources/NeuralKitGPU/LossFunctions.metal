//
//  LossFunctions.metal
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


kernel void Loss_delta(const 	device 	float*		expected			[[buffer(0)]],
					   constant			matrix3_t	&size_descriptor	[[buffer(1)]],
					   const 	device 	float*		actual				[[buffer(2)]],
					   			device	float*		loss				[[buffer(4)]],
					  			 		uint3		pos					[[thread_position_in_grid]])
{
	float ex = matrix3_get(size_descriptor, expected, pos[0], pos[1], pos[2]);
	float ac = matrix3_get(size_descriptor, actual, pos[0], pos[1], pos[2]);
	
	matrix3_set(size_descriptor, loss, pos[0], pos[1], pos[2], ex - ac);
}
