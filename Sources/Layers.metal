//
//  Layers.metal
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


float matrix_get(constant matrix_t &matrix_descriptor, const device float* matrix_store, uint column, uint row);
float matrix_get(matrix_t &matrix_descriptor, const device float* matrix_store, uint column, uint row);
void matrix_set(constant matrix_t &matrix_descriptor, device float* matrix_store, uint column, uint row, float value);
void matrix_set(matrix_t &matrix_descriptor, device float* matrix_store, uint column, uint row, float value);

float matrix3_get(constant matrix3_t &matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice);
float matrix3_get(matrix3_t &matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice);
void matrix3_set(constant matrix3_t &matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value);
void matrix3_set(matrix3_t &matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value);

/**
 
 Retrieves an element of a 2D matrix.
 
 This method assumes that the matrix values are stored in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - returns: The value stored at (column, row) in the matrix
 
 */
float matrix_get(constant matrix_t &matrix_descriptor, const device float* matrix_store, uint column, uint row)
{
	return matrix_store[matrix_descriptor.width * row + column];
}

float matrix_get(matrix_t &matrix_descriptor, const device float* matrix_store, uint column, uint row)
{
	return matrix_store[matrix_descriptor.width * row + column];
}

/**
 
 Sets an element of a 2D matrix.
 
 This method assumes that the matrix values are stored in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - parameter value: The value to which the matrix at (column, row) should be set to
 
 */
void matrix_set(constant matrix_t &matrix_descriptor, device float* matrix_store, uint column, uint row, float value)
{
	matrix_store[matrix_descriptor.width * row + column] = value;
}

void matrix_set(matrix_t &matrix_descriptor, device float* matrix_store, uint column, uint row, float value)
{
	matrix_store[matrix_descriptor.width * row + column] = value;
}


/**
 
 Retrieves an element of a 3D matrix.
 
 This method assumes that the matrix values are stored in slice major and then in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - parameter slice: The z index at which the matrix should be accessed
 - returns: The value stored at (column, row) in the matrix
 
 */
float matrix3_get(constant matrix3_t &matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice)
{
	return matrix_store[(matrix_descriptor.height * slice + row) * matrix_descriptor.width + column];
}

float matrix3_get(matrix3_t &matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice)
{
	return matrix_store[(matrix_descriptor.height * slice + row) * matrix_descriptor.width + column];
}


/**
 
 Sets an element of a 3D matrix.
 
 This method assumes that the matrix values are stored in slice major and then in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - parameter slice: The z index at which the matrix should be accessed
 - parameter value: The value to which the matrix at (column, row) should be set to
 
 */
void matrix3_set(constant matrix3_t &matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value)
{
	matrix_store[(matrix_descriptor.height * slice + row) * matrix_descriptor.width + column] = value;
}

void matrix3_set(matrix3_t &matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value)
{
	matrix_store[(matrix_descriptor.height * slice + row) * matrix_descriptor.width + column] = value;
}


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
	float result = 1 / (1 - exp(-matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2])));
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], result);
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
	float result = tanh(matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2]));
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], result);
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
	float result = fmax(matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2]), 0);
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], result);
}

kernel void PoolingLayer_forward(const	device	float*		input				[[buffer(0)]],
								 constant		matrix3_t	&input_descriptor	[[buffer(1)]],
										device	float*		output				[[buffer(2)]],
								 constant		matrix3_t	&output_descriptor	[[buffer(3)]],
												uint3		pos					[[thread_position_in_grid]])
{
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

kernel void ConvolutionLayer_forward(const	device	float*		input				[[buffer(0)]],
									 constant		matrix3_t	&input_descriptor	[[buffer(1)]],
											device	float*		output				[[buffer(2)]],
									 constant		matrix3_t	&output_descriptor	[[buffer(3)]],
									 const	device	float*		kernels				[[buffer(4)]],
									 const	device	matrix3_t*	kernel_descriptors	[[buffer(5)]],
									 constant		int			&horizontalInset	[[buffer(6)]],
									 constant		int			&verticalInset		[[buffer(7)]],
									 constant		int			&horizontalStride	[[buffer(8)]],
									 constant		int			&verticalStride		[[buffer(9)]],
													uint3		pos					[[thread_position_in_grid]])
{
	matrix3_t kernel_desc = kernel_descriptors[pos[3]];
	const device float* kernel_mat = &kernels[(kernel_desc.width + kernel_desc.height + kernel_desc.depth) * pos[3]];
	
	int sourceX = pos[0] * horizontalStride + horizontalInset;
	int sourceY = pos[1] * verticalStride + verticalInset;
	
	float convolution_sum = 0;
	
	for (uint y = 0; y < kernel_desc.height; y++)
	{
		if (((int) y) + sourceY < 0 || ((int) y) + sourceY >= (int) input_descriptor.height)
			continue;
		
		for (uint x = 0; x < kernel_desc.width; x++)
		{
			if (((int) x) + sourceX < 0 || ((int) x) + sourceX >= (int) input_descriptor.width)
				continue;
			
			for (uint z = 0; x < kernel_desc.depth; z++)
			{
				convolution_sum += matrix3_get(kernel_desc, kernel_mat, x, y, z) * matrix3_get(input_descriptor, input, sourceX, sourceY, z);
			}
		}
		
		sourceX = pos[0] * horizontalStride + horizontalInset;
		sourceY++;
	}
	
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], convolution_sum);
}
