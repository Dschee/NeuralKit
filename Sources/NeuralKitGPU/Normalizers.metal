//
//  Normalizers.metal
//  NeuralKit
//
//  Created by Palle Klewitz on 13.04.17.
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


kernel void Normalize_l1(		device	float*		weights			[[buffer(0)]],
						 constant		uint		&count			[[buffer(1)]],
						 		device	float*		gradients		[[buffer(2)]],
						 constant		float		&decay			[[buffer(3)]],
										uint		index			[[thread_position_in_grid]])
{
	if (index >= count)
		return;
	
	gradients[index] += weights[index] < 0 ? -decay : decay;
}


kernel void Normalize_l2(		device	float*		weights			[[buffer(0)]],
						 constant		uint		&count			[[buffer(1)]],
						 		device	float*		gradients		[[buffer(2)]],
						 constant		float		&decay			[[buffer(3)]],
						 uint		index			[[thread_position_in_grid]])
{
	if (index >= count)
		return;
	
	gradients[index] += weights[index] * decay;
}
