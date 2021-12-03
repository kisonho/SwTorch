//
//  Tensor.swift
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
    ///   - value: A `PythonConvertible` real value of the tensor
    ///   - dtype: the target `DType` of this `Tensor`
    public init<ValueType: PythonConvertible>(value: ValueType, dtype: DType? = nil) {
        self.tensorPtr = torch.tensor(value, dtype: dtype?.toPyType())
//        self.type = dtype
    }
    
    /// backward function
    /// - Parameters:
    ///   - gradient: An optional gradient `Tensor`
    ///   - retainGraph: A `Bool` flag of the graph used to compute the grads will be freed
    ///   - createGraph: A `Bool` flag of graph of the derivative will be constructed
    ///   - inputs: An optional `Array<Tensor>` of the gradient will be accumulated into
    public func backward(gradient: Tensor? = nil, retainGraph: Bool = false, createGraph: Bool = false, inputs: Array<Tensor>? = nil) {
        self.tensorPtr.backward(gradient, retain_graph: retainGraph, create_graph: createGraph, inputs: inputs)
    }
    
    /// Detach current tensor
    public mutating func detach() -> Tensor {
        self.tensorPtr = self.tensorPtr.detach()
        return self
    }
    
    /// Initialize a `Tensor` with all ones elements
    /// - Parameters:
    ///   - shape: An shape of `Tensor` in `Array<Int>`
    ///   - dtype: An optional `DType`
    /// - Returns: A `Tensor` with all ones in given shape and dtype
    public static func ones(_ shape: Array<Int>, dtype: DType? = nil) -> Tensor {
        return Tensor(torch.ones(shape, dtype?.toPyType()))
    }
    
    /// Initialize a `Tensor` with all zeros elements
    /// - Parameters:
    ///   - shape: An shape of `Tensor` in `Array<Int>`
    ///   - dtype: An optional `DType`
    /// - Returns: A `Tensor` with all zeros in given shape and dtype
    public static func zeros(_ shape: Array<Int>, dtype: DType? = nil) -> Tensor {
        return Tensor(torch.zeros(shape, dtype?.toPyType()))
    }
}

extension Tensor {
    /// Get the absolute value of current `Tensor`
    /// - Returns: A `Tensor` of absoluted values
    public func abs() -> Tensor {
        return Tensor(tensorPtr.abs())
    }
    
    /// Get the max index in dimension
    /// - Parameter dim: `Int` of dimension of argmax
    /// - Returns: A `Tensor` of argmax of current tensor
    public func argmax(dim: Int = 0) -> Tensor {
        return Tensor(tensorPtr.argmax(dim: dim))
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
    
    /// Check is nan in tensor
    /// - Returns: A `Tensor` of `Bool` flag if target value is nan
    public func isnan() -> Tensor {
        return Tensor(self.tensorPtr.isnan())
    }
    
    /// Calculate log of current tensor
    public func log() -> Tensor {
        return Tensor(self.tensorPtr.log())
    }
    
    /// Calculate the mean
    /// - Parameter dim: `Int` of dimension of mean
    /// - Returns: A `Tensor` of mean of current tensor
    public func mean(dim: Int = 0) -> Tensor {
        return Tensor(self.tensorPtr.mean(dim: dim))
    }
    
    /// Calculate the matrix norm or vector norm of current tensor.
    /// - Parameter dim: `Int` of dimension of mean
    /// - Returns: A `Tensor` of matrix norm of current tensor
    public func norm(dim: Int = 0) -> Tensor {
        return Tensor(self.tensorPtr.norm(dim: dim))
    }
    
    /// Reshape current tensor
    /// - Parameter shape: A target shape in `Array<Int>`
    /// - Returns: A reshaped `Tensor`
    public func reshape(_ shape: Array<Int>) -> Tensor {
        return Tensor(self.tensorPtr.reshape(shape))
    }
    
    /// Sum along dimension
    /// - Parameter dim: `Int` of dimension of sum
    public func sum(dim: Int = 0) -> Tensor {
        return Tensor(self.tensorPtr.sum(dim: dim))
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
