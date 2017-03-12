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
		let network = NeuralNetwork.init(layers: [layer], outputActivation: .linear)!
		
		let encoded = try! JSONCoder.encode(network)

		do
		{
			let decoded: NeuralNetwork = try JSONCoder.decode(encoded)
			print(String(reflecting: decoded))
		}
		catch
		{
			print(error)
		}
	}
}
