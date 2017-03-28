//
//  SerializationTests.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 26.02.17.
//
//

import Foundation
import XCTest
@testable import NeuralKit

class SerializationTests: XCTestCase
{
	func testSerialization()
	{
		let matrix = RandomWeightMatrix(width: 10, height: 10)
		let layer = FullyConnectedLayer(weights: matrix, activationFunction: .tanh)
		let network = FeedForwardNeuralNetwork(layers: [layer], outputActivation: .linear)!
		
		let encoded = try! JSONCoder.encode(network)

		do
		{
			let decoded: FeedForwardNeuralNetwork = try JSONCoder.decode(encoded)
			print(String(reflecting: decoded))
		}
		catch
		{
			print(error)
		}
	}
}
