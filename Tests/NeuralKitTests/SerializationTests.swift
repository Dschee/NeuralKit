//
//  SerializationTests.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 26.02.17.
//
//

import Foundation
import XCTest
import SwiftyJSON
@testable import NeuralKit

class SerializationTests: XCTestCase
{
	func testSerialization()
	{
		let matrix = RandomWeightMatrix(width: 10, height: 10)
		let data = matrix.serialize()
		
		print(data)
		
		guard let deserialized = Matrix(data) else
		{
			XCTFail()
			return
		}
		XCTAssertEqual(matrix.width, deserialized.width)
		XCTAssertEqual(matrix.height, deserialized.height)
		XCTAssertEqual(matrix.values, deserialized.values)
	}
}
