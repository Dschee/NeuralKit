//
//  NeuralKit.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 19.02.17.
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


import Foundation

public struct NeuralNetwork
{
	public internal(set) var layers: [NeuralLayer]
	
	public init(layers: [NeuralLayer])
	{
		for i in 1 ..< layers.count
		{
			precondition(layers[i-1].outputSize == layers[i].inputSize, "Layers \(i-1) and \(i) must have matching output and input size")
		}
		self.layers = layers
	}
	
	public func feedForward(_ sample: Matrix3) -> Matrix3
	{
		return layers.reduce(sample)
		{
			sample, layer in
			layer.forward(sample)
		}
	}
	
	
}
