//
//  MNISTTest.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 25.02.17.
//
//

import XCTest
@testable import NeuralKit

class MNISTTest: XCTestCase
{
	func images(from path: String) -> ([TrainingSample], [TrainingSample])
	{
		func readSamples(from bytes: [UInt8], labels: [UInt8], count: Int) -> [TrainingSample]
		{
			let imageOffset = 16
			let labelOffset = 8
			
			let imageWidth = 28
			let imageHeight = 28
			
			var samples:[TrainingSample] = []
			
			for i in 0 ..< count
			{
				let offset = imageOffset + imageWidth * imageHeight * i
				let pixelData = bytes[offset ..< (offset + imageWidth * imageHeight)]
					.map{Float($0)}
					.map{$0/128}
					.map{$0-1}
				
				let sampleMatrix = Matrix3(values: pixelData, width: imageWidth, height: imageHeight, depth: 1)
				
				let label = Int(labels[labelOffset + i])
				
				let sample = TrainingSample(values: sampleMatrix, outputCount: 10, targetIndex: label)
			}
		}
		
		guard
			let trainingBytes = try? Data(contentsOf: URL(fileURLWithPath: path + "train-images-idx3-ubyte")).bytes,
			let trainingLabels = try? Data(contentsOf: URL(fileURLWithPath: path + "train-labels-idx1-ubyte")).bytes,
			let testingBytes = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-images-idx3-ubyte")).bytes,
		let testingLabels = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-labels-idx1-ubyte")).bytes
		else
		{
			return ([],[])
		}
		
		let trainingSampleCount = 60_000
		let testingSampleCount = 10_000
		
		let trainingSamples = readSamples(from: trainingBytes, labels: trainingLabels, count: trainingSampleCount)
		let testinSamples = readSamples(from: testingBytes, labels: testingLabels, count: testingSampleCount)
		
		return ([],[])
	}
}
