//
//  ImageConversion.swift
//  NeuralKit
//
//  Created by Palle Klewitz on 02.02.16.
//  Copyright © 2016 - 2017 Palle Klewitz.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is furnished
//  to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

import Foundation
import CoreGraphics
import QuartzCore
import Accelerate
import Cocoa
import MatrixVector


/**

Filter for image preprocessing

A CGImageFilter is a function with a CGImage
as an argument, which returns a modified CGImage.

If the image should not be modified, CGImageFilterIdentity can be used.

*/
public typealias CGImageFilter = (CGImage) -> CGImage


/**

Appends the right CGImageFilter to the left CGImageFilter.

When using the returned CGImageFilter, the filter, which was passed
left to the '+'-operator will be evaluated first. Then the filter
passed on the right will be evaluated with the result of the left filter.

The '+'-operator for CGImageFilters is associative: (a + b) + c = a + (b + c)

The '+'-operator is non-commutative: a + b ≠ b + a

- parameter left: CGImageFilter which should be evaluated first

- parameter right: CGImageFilter which should be evaluated second

- returns: A newly created CGImageFilter-function consisting of the two filter functions.

*/
public func + (left: @escaping CGImageFilter, right: @escaping CGImageFilter) -> CGImageFilter
{
	return CGImageFilterAppendFilter(filter: right, toFilter: left)
}


/**

Appends the first CGImageFilter to the second CGImageFilter.

When using the returned CGImageFilter, the filter, which was passed
second to the '+'-operator will be evaluated first.
Then the filter passed second will be evaluated with the result of the left filter.

- parameter second: CGImageFilter which should be evaluated last

- parameter first: CGImageFilter which should be evaluated first

- returns: A newly created CGImageFilter-function consisting of the two filter functions.

*/
public func CGImageFilterAppendFilter(filter second: @escaping CGImageFilter, toFilter first: @escaping CGImageFilter) -> CGImageFilter
{
	return { return second(first($0)) }
}


/**

Identity filter

returns the image without modifying it.

The identity filter is neutral: filter + CGImageFilterIdentity == CGImageFilterIdentity + filter == filter

*/
public let CGImageFilterIdentity: CGImageFilter =
{ image in
	return image
}


/**

Protocol for retrieving pixel data from an object.

Includes three types of data which must be provided:

- pixelData: the raw pixel data in 8 bit RGBA format

- normalizedPixelData: the pixel data transformed into a range between -1.0 and 1.0

- vImage: the pixel data as a vImage for accelerated image processing

*/
public protocol PixelDataConvertible
{
	
	/**
	
	pixel data of the image in raw 8 bit RGBA format
	
	the data will be in unsigned 8 bit integer format.
	
	Values must be in the range 0 to 255.
	
	- returns: RGBA8888 pixel data
	
	*/
	var pixelData: [UInt8]? { get }
	
	
	/**
	
	normalized pixel data.
	
	The data will be in double precision floating point format.
	
	Values should be in the range -1.0 to 1.0.
	
	- returns: normalized pixel data
	
	*/
	var normalizedPixelData: [Float]? { get }
	
	
	/**
	
	vImage buffer containing the pixel data
	
	Used for accelerated image processing
	
	- returns: vImage buffer containing the pixel data of the current object.
	
	*/
	var vImage:vImage_Buffer? { get }
	
}


/**

Extension for CGImage scaling

Scales an image to a desired size using CoreGraphics.

*/
public extension CGImage
{
	
	/**
	
	Scales the current bitmap image to the desired size.
	
	Quartz scales the image—disproportionately,
	if necessary—to fit the bounds specified by the targetSize parameter.
	
	Creates a CGImage object that contains the original image
	or nil if the image is not created.
	
	- parameter targetSize: Size of the output image
	
	- returns: An image with the scaled contents of this image and a size of targetSize.
	
	*/
	public func scaled(toSize targetSize: CGSize) -> CGImage?
	{
		let context = CGContext(
			data: nil,
			width: Int(targetSize.width),
			height: Int(targetSize.height),
			bitsPerComponent: self.bitsPerComponent,
			bytesPerRow: self.bytesPerRow,
			space: self.colorSpace!,
			bitmapInfo: self.bitmapInfo.rawValue)
		
		context?.draw(self, in: CGRect(origin: CGPoint.zero, size: targetSize))
		
		return context?.makeImage()
	}
	
}


/**

Let CGImage conform to the PixelData protocol

Pixel data, normalized pixel data and
a vImage_buffer can now be retrieved from a CGImage.

*/
extension CGImage : PixelDataConvertible
{
	
	/**
	
	pixel data of the image in raw 8 bit RGBA format
	
	the data will be in unsigned 8 bit integer format.
	
	Values must be in the range 0 to 255.
	
	- returns: RGBA8888 pixel data
	
	*/
	public var pixelData: [UInt8]?
	{
		// 8 bit R, G, B and A channels
		let bytesPerPixel = 4
		let colorSpace = CGColorSpaceCreateDeviceRGB()
		
		//allocate buffer to store pixel data
		var pixelData = Array<UInt8>(repeating: 0, count: width * height * bytesPerPixel)
		
		//info requires a 32 bit unsigned integer value.
		let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
		
		//creating bitmap context with the allocated buffer as target
		let context = CGContext(data: &pixelData, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: info)
		
		//drawing the CGImage onto the bitmap context
		context?.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))
		
		//the buffer now contains the pixel data of the current image
		return pixelData
	}
	
	
	/**
	
	normalized pixel data.
	
	The data will be in double precision floating point format.
	
	Values should be in the range -1.0 to 1.0.
	
	- returns: normalized pixel data
	
	*/
	public var normalizedPixelData: [Float]?
	{
		//mapping from UInt8 (0 ... 255) to Float (0.0 ... 1.0)
		return self.pixelData?.map { Float($0) / 255 }
	}
	
	
	/**

	Color matrix of the image.
	
	The width and height of the matrix is equal to the width and height of the image.
	The depth of the matrix is 3, where the z index 0 contains the red color data, 1 green and 2 blue.
	
	The alpha channel of the image is ignored.
	
	- returns: Color matrix of the image.
	
	*/
	public var colorMatrix: Matrix3?
	{
		var matrix = Matrix3(repeating: 0, width: self.width, height: self.height, depth: 3)
		
		guard let pixelData = normalizedPixelData else
		{
			return nil
		}
		
		for (x,y,z) in matrix.indices
		{
			matrix[x,y,z] = pixelData[x*4 + y*4*width + z]
		}
		
		return matrix
	}
	
	
	/**

	Normalized greyscale pixel data.
	
	The greyscale representation is calculated by calculating the root of 
	the summed squares of the components, multiplying it by the alpha value
	and dividing it by the root of 3.
	
	*/
	public var normalizedGreyscalePixelData: [Float]?
	{
		guard let pixelData = self.normalizedPixelData else { return nil }
		var result = [Float](repeating: 0, count: pixelData.count / 4)
		for i in result.indices
		{
			result[i] = (pixelData[4*i] * 0.2990 + pixelData[4*i+1] * 0.5870 + pixelData[4*i+2] * 0.1140) * pixelData[4*i+3]
			//result[i] = sqrt(pixelData[4*i] * pixelData[4*i] + pixelData[4*i+1] * pixelData[4*i+1] + pixelData[4*i+2] * pixelData[4*i+2]) * pixelData[4*i+3] / sqrtf(3)
		}
		return result
	}
	
	
	/**
	
	Greyscale matrix of the image.
	
	The width and height of the matrix is equal to the width and height of the image.
	The depth of the matrix is 1.
	
	The alpha channel of the image is ignored.
	
	- returns: Greyscale matrix of the image.
	
	*/
	public var greyscaleMatrix: Matrix3?
	{
		var matrix = Matrix3(repeating: 0, width: self.width, height: self.height, depth: 1)
		
		guard let pixelData = normalizedGreyscalePixelData else
		{
			return nil
		}
		
		for (x,y,z) in matrix.indices
		{
			matrix[x,y,z] = pixelData[x + y*width]
		}
		
		return matrix
	}
	
	
	/**
	
	vImage buffer containing the pixel data
	
	Used for accelerated image processing
	
	- returns: vImage buffer containing the pixel data of the current object.
	
	*/
	public var vImage:vImage_Buffer?
	{
		let colorSpace = CGColorSpaceCreateDeviceRGB()
		
		//info requires a 32 bit unsigned integer value.
		let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
		
		//creating vImage format
		var format = vImage_CGImageFormat(
			bitsPerComponent:	UInt32(bitsPerComponent),
			bitsPerPixel:		UInt32(bitsPerPixel),
			colorSpace:			Unmanaged.passUnretained(colorSpace),
			bitmapInfo:			CGBitmapInfo(rawValue: bitmapInfo),
			version:			0,
			decode:				nil,
			renderingIntent:	CGColorRenderingIntent.defaultIntent)
		
		//creating image buffer
		var buffer = vImage_Buffer()
		
		//creating buffer with current image
		vImageBuffer_InitWithCGImage(&buffer, &format, nil, self, UInt32(kvImageNoFlags))
		return buffer
	}
	
}


/**

Adds an initializer for loading CGImages from files or raw data.

*/
public extension CGImage
{
	
	/**

	Initializes CGImage instance by loading an image from the specified file path.
	
	- parameter url: Path to the file containing the image
	
	- returns: An image containing data from the specified image file.
	
	*/
	public static func load(from url: URL) -> CGImage?
	{
		guard let nsImage = NSImage(contentsOf: url) else { return nil }
		var imageRect = CGRect(x: 0, y: 0, width: nsImage.size.width, height: nsImage.size.height)
		let image = nsImage.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)
		return image
	}
	
	
	/**

	Initializes a CGImage instance from the given raw data.
	
	- parameter from: Source data
	
	- parameter width: Width of the image
	
	- parameter height: Height of the image
	
	- parameter minValue: Optional minimum value, which should be treated as black.
	If no value is specified, the minimum value of the vector is used.
	
	- parameter maxValue: Optional maximum value, which should be treated as white.
	If no value is specified, the maximum value of the vector is used.
	
	- returns: Image created from the data contained in the input vector
	
	*/
	public static func make(from: [Float], width: Int, height: Int, minValue: Float? = nil, maxValue: Float? = nil) -> CGImage?
	{
		let minValue = minValue ?? min(from)
		let maxValue = maxValue ?? max(from)
		
		let maxMagnitude = max(-minValue, maxValue)
		
		let colorSpace = CGColorSpaceCreateDeviceGray()
		var bytes = from.map{($0 + maxMagnitude) / max(2 * maxMagnitude, Float.leastNonzeroMagnitude) * 255}.map{UInt8($0)}
		
//		let image = withUnsafeMutablePointer(to: &bytes) { (pointer) -> CGImage? in
//			let rawPointer = UnsafeMutableRawPointer(pointer)
//			let ctx = CGContext(data: rawPointer, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width, space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue)
//			return ctx?.makeImage()
//		}
		
		guard let context = CGContext(
			data: nil,
			width: width,
			height: height,
			bitsPerComponent: 8,
			bytesPerRow: width,
			space: colorSpace,
			bitmapInfo: CGImageAlphaInfo.none.rawValue
		)
		else
		{
			print("Error: could not create context.")
			return nil
		}
		
		guard let data = context.data?.assumingMemoryBound(to: UInt8.self)
		else
		{
			print("Error: Could not retrieve data of context.")
			return nil
		}
		
		for index in 0 ..< (width * height)
		{
			data[index] = bytes[index]
		}
		
		return context.makeImage()
	}
	
	
	public static func make(from matrix: Matrix, minValue: Float? = nil, maxValue: Float? = nil) -> CGImage?
	{
		return self.make(from: matrix.values, width: matrix.width, height: matrix.height, minValue: minValue, maxValue: maxValue)
	}
	
	
	public static func make(from matrix: Matrix3, minValue: Float? = nil, maxValue: Float? = nil) -> [CGImage]
	{
		var result: [CGImage] = []
		
		if matrix.width == 1 && matrix.height == 1
		{
			return self.make(from: matrix.values, width: matrix.values.count, height: 1, minValue: minValue, maxValue: maxValue).flatMap{[$0]} ?? []
		}
		
		for z in 0 ..< matrix.depth
		{
			guard
				let image = self.make(
					from: Matrix(
						matrix[
							x: 0,
							y: 0,
							z: z,
							width: matrix.width,
							height: matrix.height,
							depth: 1
						]
					),
					minValue: minValue,
					maxValue: maxValue
				)
			else
			{
				continue
			}
			result.append(image)
		}
		
		return result
	}
	
	@discardableResult
	public func write(to url: URL) -> Bool
	{
		guard let destination = CGImageDestinationCreateWithURL(url as CFURL, kUTTypePNG, 1, nil) else
		{
			return false
		}
		CGImageDestinationAddImage(destination, self, nil)
		return CGImageDestinationFinalize(destination)
	}
}
