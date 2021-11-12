//
//  File.swift
//  
//
//  Created by Kison Ho on 10/28/21.
//

import PythonKit

/// The PyTorch DType mapping
public enum DType {
    case bool
    case complex32, complex64, complex128
    case int8, int16, int32, int64
    case float16, float32, float64
    case uint8
    
    /// to pytorch dtype
    /// - Returns: A `torch.<dtype>` for the mapping type in `PythonObject`
    fileprivate func toPyType() -> PythonObject {
        // switch type
        switch self {
        case .bool: return torch.bool
            
        case .complex32: return torch.complex32
        case .complex64: return torch.complex64
        case .complex128: return torch.complex128
            
        case .int8: return torch.int8
        case .int16: return torch.int16
        case .int32: return torch.int32
        case .int64: return torch.int64
            
        case .float16: return torch.float16
        case .float32: return torch.float32
        case .float64: return torch.float64
            
        case .uint8: return torch.uint8
        }
    }
}

/// main tensor struct
public struct Tensor {
    /// Get current tensor shape
    public var shape: Array<Int> { get {
        return Array(tensorPtr.shape)!
    }}
    
    /// The tensor pointer that torch.Tensor stored
    fileprivate var tensorPtr: PythonObject
    
    /// Get current tensor type
//    public var type: DType
    
    /// Constructor
    /// - Parameters:
    ///   - value: A PythonConvertible real value of the tensor
    ///   - shape:
    public init<ValueType: PythonConvertible>(value: ValueType, dtype: DType? = nil) {
        self.tensorPtr = torch.tensor(value, dtype: dtype?.toPyType())
//        self.type = dtype
    }
    
    /// backward function
    public func backward() {
        self.tensorPtr.backward()
    }
}

extension Tensor {
    /// Get the absolute value of current `Tensor`
    /// - Returns: A `Tensor` of absoluted values
    public func abs() -> Tensor {
        return Tensor(tensorPtr.abs())
    }
    
    /// Get the max index in axis
    /// - Parameter axis: `Int` of axis of argmax
    /// - Returns: A `Tensor` of argmax of current tensor
    public func argmax(axis: Int? = nil) -> Tensor {
        return Tensor(tensorPtr.argmax(axis: axis))
    }
    
    /// Concat multiple tensors
    /// - Parameters:
    ///   - tensors: The `Array` of `Tensor` to concat
    ///   - dim: The `Int` of concat dimention
    /// - Returns: A concatenated `Tensor `
    public static func concat(_ tensors: Array<Tensor>, dim: Int = 0) -> Tensor {
        return Tensor(torch.cat(tensors, dim: dim))
    }
    
    /// Get the equal value among values of current `Tensor` and another `Tensor`
    /// - Parameter other: Another `Tensor` to be compared
    /// - Returns: A `Tensor` of Bool on all values
    public func equal(_ other: Tensor) -> Tensor {
        return Tensor(torch.eq(self, other))
    }
    
    /// Flattens a tensor according to dims
    /// - Parameters:
    ///   - startDim: The `Int` of first dim to flatten
    ///   - endDim: The `Int` of last dim to flatten
    /// - Returns: A flattened `Tensor`
    public func flatten(startDim: Int = 0, endDim: Int = -1) -> Tensor {
        return Tensor(tensorPtr.flatten(startDim, endDim))
    }
    
    /// calculate the mean
    /// - Parameter axis: `Int` of axis of mean
    /// - Returns: A `Tensor` of mean of current tensor
    public func mean(axis: Int? = nil) -> Tensor {
        return Tensor(self.tensorPtr.mean(axis: axis))
    }
    
    /// Reshape current tensor
    /// - Parameter shape: A target shape in `Array<Int>`
    /// - Returns: A reshaped `Tensor`
    public func reshape(_ shape: Array<Int>) -> Tensor {
        return Tensor(self.tensorPtr.reshape(shape))
    }
    
    /// Sum along the axis
    /// - Parameter axis: `Int` of axis of sum
    public func sum(axis: Int? = nil) -> Tensor {
        return Tensor(self.tensorPtr.sum(axis: axis))
    }
    
    /// Cast current tensor to a target dtype
    /// - Parameter dtype: A `DType` enum of target dtype
    /// - Returns: A converted `Tensor`
    public func to(dtype: DType) -> Tensor {
        return Tensor(self.tensorPtr.to(dtype.toPyType()))
    }
}

extension Tensor: ConvertibleFromPython {
    /// Constructor
    /// - Parameters:
    ///   - object: The `PythonObject` in `torch.Tensor` that convert from
    public init(_ object: PythonObject) {
        tensorPtr = object
    }
}

extension Tensor: Comparable, Equatable {
    public static func < (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr < rhs.tensorPtr
    }
    
    public static func <= (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr <= rhs.tensorPtr
    }
    
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return Bool(torch.equal(lhs.tensorPtr, rhs.tensorPtr))!
    }
    
    public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
        return !Bool(torch.equal(lhs.tensorPtr, rhs.tensorPtr))!
    }
    
    public static func > (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr > rhs.tensorPtr
    }
    
    public static func >= (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr >= rhs.tensorPtr
    }
}

extension Tensor: CustomStringConvertible {
    public var description: String {
        return tensorPtr.description
    }
}

extension Tensor: DeviceMovable {
    public mutating func to(_ device: Device, id: Int? = nil) {
        let d = id == nil ? device.rawValue : "\(device.rawValue):\(id!)"
        self.tensorPtr = self.tensorPtr.to(torch.device(d))
    }
}

extension Tensor: Numeric {
    public typealias IntegerLiteralType = Int
    
    public typealias Magnitude = Tensor
    
    public var magnitude: Tensor {
        return Tensor(torch.abs(tensorPtr))
    }
    
    public init?<T>(exactly source: T) where T : BinaryInteger {
        self.tensorPtr = torch.Tensor(Int(source))
//        self.type = .int64
    }
    
    public init(integerLiteral value: Int) {
        self.tensorPtr = torch.Tensor(value)
//        self.type = .int64
    }
    
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr + rhs.tensorPtr)
    }
    
    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr - rhs.tensorPtr)
    }
    
    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr * rhs.tensorPtr)
    }
    
    public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr / rhs.tensorPtr)
    }
    
    public static func ^ (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(torch.pow(lhs, rhs))
    }
    
    public static func += (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs + rhs
    }
    
    public static func -= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs - rhs
    }
    
    public static func *= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs * rhs
    }
    
    public static func /= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs / rhs
    }
}

extension Tensor: PythonConvertible {
    /// The torch.Tensor python object
    public var pythonObject: PythonObject { get {
        return self.tensorPtr
    }}
}

extension Tensor: Sequence {
    public typealias Element = PythonObject
    
    public typealias Iterator = PythonObject.Iterator
    
    public func makeIterator() -> PythonObject.Iterator {
        tensorPtr.makeIterator()
    }
}

extension Array where Element: PythonConvertible {
    public init?(_ tensor: Tensor) {
        self = []
        for t in tensor.tensorPtr {
            append(t as! Element)
        }
    }
}

extension Double {
    /// Convert a `Tensor` to a `Double`
    public init?(_ tensor: Tensor) {
        self = Double(tensor.tensorPtr)!
    }
}

extension Float {
    /// Convert a `Tensor` to a `Float`
    public init?(_ tensor: Tensor) {
        guard let v = Float(tensor.tensorPtr) else { return nil }
        self.init(v)
    }
}

extension Int {
    /// Convert a `Tensor` to an `Int`
    public init?(_ tensor: Tensor) {
        guard let v = Int(tensor.tensorPtr) else { return nil }
        self.init(v)
    }
}
