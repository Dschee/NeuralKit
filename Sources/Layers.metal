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


//class MutableMatrix3
//{
//	device float* values;
//	
//public:
//	uint width;
//	uint height;
//	uint depth;
//	
//	float getValue(uint x, uint y, uint z)
//	{
//		return values[(int)((height * z + y) * width + x)];
//	}
//	
//	MutableMatrix3(device float* values, matrix3_t descriptor)
//	{
//		this->values = values;
//		this->width = descriptor.width;
//		this->height = descriptor.height;
//		this->depth = descriptor.depth;
//	}
//};

float matrix_get(matrix_t matrix_descriptor, const device float* matrix_store, uint column, uint row);

void matrix_set(matrix_t matrix_descriptor, device float* matrix_store, uint column, uint row, float value);


float matrix3_get(matrix3_t matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice);

void matrix3_set(matrix3_t matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value);

/**
 
 Retrieves an element of a 2D matrix.
 
 This method assumes that the matrix values are stored in row major data ordering
 
 - parameter descriptor: Descriptor storing the dimensions of the matix
 - parameter matrix_store: The contents of the matrix
 - parameter column: The column at which the matrix should be accessed
 - parameter row: The row at which the matrix should be accessed
 - returns: The value stored at (column, row) in the matrix
 
 */
float matrix_get(matrix_t matrix_descriptor, const device float* matrix_store, uint column, uint row)
{
	return matrix_store[(int)(matrix_descriptor.width * row + column)];
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
void matrix_set(matrix_t matrix_descriptor, device float* matrix_store, uint column, uint row, float value)
{
	matrix_store[(int)(matrix_descriptor.width * row + column)] = value;
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
float matrix3_get(matrix3_t matrix_descriptor, const device float* matrix_store, uint column, uint row, uint slice)
{
	return matrix_store[(int)((matrix_descriptor.height * slice + row) * matrix_descriptor.width + column)];
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
void matrix3_set(matrix3_t matrix_descriptor, device float* matrix_store, uint column, uint row, uint slice, float value)
{
	matrix_store[(int)((matrix_descriptor.height * slice + row) * matrix_descriptor.width + column)] = value;
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
											  		device	float*		weight_delta				[[buffer(8)]], // skip weight delta descriptor
											  constant		float		&learning_rate				[[buffer(10)]],
											  constant		float		&momentum					[[buffer(11)]],
											  constant		float		&decay						[[buffer(12)]],
											  				uint		column						[[thread_position_in_grid]])
{
	if (column > input_descriptor.depth)
		return;
	
	// Don't calculate gradient for bias
	if (column != weight_descriptor.width - 1)
	{
		float grad = 0;
		
		for (uint row = 0; row < next_gradient_descriptor.depth; row++)
		{
			grad += matrix3_get(next_gradient_descriptor, next_gradient, 0, 0, row) * matrix_get(weight_descriptor, weights, column, row);
		}
		
		matrix3_set(input_descriptor, gradient, 0, 0, column, grad);
	}
	
	float in = (column == (weight_descriptor.width - 1)) ? 1 : matrix3_get(input_descriptor, input, 0, 0, column);
	
	if (momentum != 0)
	{
		for (uint row = 0; row < weight_descriptor.height; row++)
		{
			float delta = matrix3_get(next_gradient_descriptor, next_gradient, 0, 0, row) * in;
			float previous_delta = matrix_get(weight_descriptor, weight_delta, column, row);
			float update_delta = learning_rate * delta + momentum * previous_delta;
			
			matrix_set(weight_descriptor, weight_delta, column, row, update_delta);
			
			float new_weight = matrix_get(weight_descriptor, weights, column, row) + update_delta;
			matrix_set(weight_descriptor, weights, column, row, new_weight);
		}
	}
	else
	{
		for (uint row = 0; row < weight_descriptor.height; row++)
		{
			float delta = matrix3_get(next_gradient_descriptor, next_gradient, 0, 0, row) * in;
			float new_weight = delta * learning_rate + matrix_get(weight_descriptor, weights, column, row);
			
			matrix_set(weight_descriptor, weights, column, row, new_weight);
		}
	}
	
	if (decay != 0)
	{
		float scale = 1 - decay;
		
		for (uint row = 0; row < weight_descriptor.height; row++)
		{
			float current = matrix_get(weight_descriptor, weights, column, row);
			matrix_set(weight_descriptor, weights, column, row, current * scale);
		}
	}
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
	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
		return;
	
	float result = 1 / (1 - exp(-matrix3_get(input_descriptor, input, pos[0], pos[1], pos[2])));
	matrix3_set(output_descriptor, output, pos[0], pos[1], pos[2], result);
}


kernel void NonlinearityLayer_backpropagate_sigmoid(const	device	float*		next_gradient				[[buffer(0)]],
													constant		matrix3_t	&next_gradient_descriptor	[[buffer(1)]],
															device	float*		gradient					[[buffer(2)]], // Skip gradient descriptor
																	uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= next_gradient_descriptor.width || pos[1] >= next_gradient_descriptor.height || pos[2] >= next_gradient_descriptor.depth)
		return;
	
	float grad = matrix3_get(next_gradient_descriptor, next_gradient, pos[0], pos[1], pos[2]);
	grad *= grad * (1 - grad);
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
														device	float*		gradient					[[buffer(2)]], // Skip gradient descriptor
																uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= next_gradient_descriptor.width || pos[1] >= next_gradient_descriptor.height || pos[2] >= next_gradient_descriptor.depth)
		return;
	
	float grad = matrix3_get(next_gradient_descriptor, next_gradient, pos[0], pos[1], pos[2]);
	grad = grad * (1 - grad * grad);
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
														device	float*		gradient					[[buffer(2)]], // Skip gradient descriptor
																uint3		pos							[[thread_position_in_grid]])
{
	if (pos[0] >= next_gradient_descriptor.width || pos[1] >= next_gradient_descriptor.height || pos[2] >= next_gradient_descriptor.depth)
		return;
	
	float grad = matrix3_get(next_gradient_descriptor, next_gradient, pos[0], pos[1], pos[2]);
	grad = fmax(grad, 0);
	matrix3_set(next_gradient_descriptor, gradient, pos[0], pos[1], pos[2], grad);
}


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
//	if (pos[0] >= output_descriptor.width || pos[1] >= output_descriptor.height || pos[2] >= output_descriptor.depth)
//		return;
	
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

kernel void ConvolutionLayer_backpropagate(const	device	float*		input						[[buffer(0)]],
										   constant			matrix3_t	&input_descriptor			[[buffer(1)]],
													device	float*		next_gradient				[[buffer(2)]],
										   constant			matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
										   device			float*		gradient					[[buffer(4)]], // Skip gradient descriptor
										   const	device	float*		kernels						[[buffer(6)]],
										   	 	 	device	matrix3_t	&kernel_descriptor			[[buffer(7)]],
													device	float*		bias_values					[[buffer(8)]],
													device	float*		kernel_deltas				[[buffer(9)]], // Skip kernel deltas descriptor
													device	float*		bias_deltas					[[buffer(11)]],
										   constant			int			&horizontal_inset			[[buffer(12)]],
										   constant			int			&vertical_inset				[[buffer(13)]],
										   constant			int			&horizontal_stride			[[buffer(14)]],
										   constant			int			&vertical_stride			[[buffer(15)]],
										   constant			float		&learning_rate				[[buffer(16)]],
										   constant			float		&momentum					[[buffer(17)]],
										   constant			float		&decay						[[buffer(18)]],
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

kernel void ConvolutionLayer_adjust_weights(const	device	float*		input						[[buffer(0)]],
											constant		matrix3_t	&input_descriptor			[[buffer(1)]],
													device	float*		next_gradient				[[buffer(2)]],
											constant		matrix3_t	&next_gradient_descriptor	[[buffer(3)]],
											device			float*		gradient					[[buffer(4)]], // Skip gradient descriptor
													device	float*		kernels						[[buffer(6)]],
													device	matrix3_t	&kernel_descriptor			[[buffer(7)]],
													device	float*		bias_values					[[buffer(8)]],
													device	float*		kernel_deltas				[[buffer(9)]], // Skip kernel deltas descriptor
													device	float*		bias_deltas					[[buffer(11)]],
											constant		int			&horizontal_inset			[[buffer(12)]],
											constant		int			&vertical_inset				[[buffer(13)]],
											constant		int			&horizontal_stride			[[buffer(14)]],
											constant		int			&vertical_stride			[[buffer(15)]],
											constant		float		&learning_rate				[[buffer(16)]],
											constant		float		&momentum					[[buffer(17)]],
											constant		float		&decay						[[buffer(18)]],
															uint3		pos							[[thread_position_in_grid]]) // Position in convolution kernels
{
	if (pos[0] >= kernel_descriptor.width || pos[1] >= kernel_descriptor.height || pos[2] >= kernel_descriptor.depth)
		return;
	
	matrix3_t kernel_desc = kernel_descriptor;
	kernel_desc.depth = input_descriptor.depth;
	
	const uint outputZ = pos[2] / input_descriptor.depth;
	const uint inputZ = pos[2] % input_descriptor.depth;
	
	float weightGradient = 0;
	float biasGradient = 0;
	
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
			biasGradient += next_grad; // * 1 (bias neurons always output 1)
		}
	}
	
	if (momentum != 0)
	{
		float delta = weightGradient * learning_rate + matrix3_get(kernel_descriptor, kernel_deltas, pos[0], pos[1], pos[2]) * momentum;
		matrix3_set(kernel_descriptor, kernel_deltas, pos[0], pos[1], pos[2], delta);
		float current = matrix3_get(kernel_descriptor, kernels, pos[0], pos[1], pos[2]);
		matrix3_set(kernel_descriptor, kernels, pos[0], pos[1], pos[2], current + delta);
		
		float biasDelta = biasGradient * learning_rate + bias_deltas[pos[2]] * momentum;
		bias_deltas[pos[2]] = biasDelta;
		bias_values[pos[2]] += biasDelta;
	}
	else
	{
		float delta = weightGradient * learning_rate;
		float current = matrix3_get(kernel_descriptor, kernels, pos[0], pos[1], pos[2]);
		matrix3_set(kernel_descriptor, kernels, pos[0], pos[1], pos[2], current + delta);
		
		bias_values[pos[2]] += biasGradient * learning_rate;
	}
	
	if (decay != 0)
	{
		float current = matrix3_get(kernel_descriptor, kernels, pos[0], pos[1], pos[2]);
		matrix3_set(kernel_descriptor, kernels, pos[0], pos[1], pos[2], current * (1 - decay));
		
		bias_values[pos[2]] *= (1 - decay);
	}
}
