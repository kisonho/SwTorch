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
    case qint4, qint8, qfint32, quint4x2
    
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
            
        case .qint4: return torch.qint4
        case .qint8: return torch.qint8
        case .qfint32: return torch.qfint32
        case .quint4x2: return torch.quint4x2
        }
    }
    
    /// Convert a PyTorch dtype into `DType`
    /// - Parameter pyType: The PyTorch dtype in `PythonObject`
    /// - Returns: A `DType` that mapped with given pyType
    fileprivate func fromPyType(pyType: PythonObject) -> DType? {
        switch pyType {
        case torch.bool: return .bool
            
        case torch.complex32: return .complex32
        case torch.complex64: return .complex64
        case torch.complex128: return .complex128
            
        case torch.int8: return .int8
        case torch.int16: return .int16
        case torch.int32: return .int32
        case torch.int64: return .int64
            
        case torch.float16: return .float16
        case torch.float32: return .float32
        case torch.float64: return .float64
            
        case torch.uint8: return .uint8
            
        case torch.qint4: return .qint4
        case torch.qint8: return .qint8
        case torch.qfint32: return .qfint32
        case torch.quint4x2: return .quint4x2
        default: return nil
        }
    }
}

/// main tensor struct
public struct Tensor {
    /// The tensor pointer that torch.Tensor stored
    fileprivate var tensorPtr: PythonObject
    
    /// Constructor
    /// - Parameters:
    ///   - value: A PythonConvertible real value of the tensor
    ///   - shape:
    public init<ValueType: PythonConvertible>(_ value: ValueType, dtype: DType = .float32) {
        self.tensorPtr = torch.Tensor(value, dtype: dtype.toPyType())
    }
    
    /// backward function
    public func backward() {
        self.tensorPtr.backward()
    }
}

extension Tensor: ConvertibleFromPython {
    /// Constructor
    /// - Parameters:
    ///   - object: The `PythonObject` in `torch.Tensor` that convert from
    public init?(_ object: PythonObject) {
        tensorPtr = torch.Tensor(object)
    }
    
    /// Get the max index in axis
    /// - Parameter axis: `Int` of axis of argmax
    /// - Returns: A `Tensor` of argmax of current tensor
    public func argmax(axis: Int? = nil) -> Tensor {
        return Tensor(tensorPtr.argmax(axis: axis))!
    }
    
    /// calculate the mean
    /// - Returns: A `Tensor` of mean of current tensor
    public func mean() -> Tensor {
        return Tensor(self.tensorPtr.mean())!
    }
    
    /// Cast current tensor to a target dtype
    /// - Parameter dtype: A `DType` enum of target dtype
    /// - Returns: A converted `Tensor`
    public func to(dtype: DType) -> Tensor {
        return Tensor(self.tensorPtr.to(dtype.toPyType()))
    }
}

extension Tensor: CustomStringConvertible {
    /// A textual description of this `Tensor`
    public var description: String {
        return String(tensorPtr)!
    }
}

extension Tensor: DeviceMovable {
    /// Move current module to a target device
    /// - Parameter device: A `String` of device
    public func to(_ device: Device) {
        self.tensorPtr.to(torch.device(device.rawValue))
    }
}

extension Tensor: Equatable, Comparable {
    public static func < (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr < rhs.tensorPtr
    }
    
    public static func <= (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr <= rhs.tensorPtr
    }
    
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr == rhs.tensorPtr
    }
    
    public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr != rhs.tensorPtr
    }
    
    public static func > (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr > rhs.tensorPtr
    }
    
    public static func >= (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.tensorPtr >= rhs.tensorPtr
    }
}

extension Tensor: Numeric {
    public init?<T>(exactly source: T) where T : BinaryInteger {
        self.tensorPtr = torch.Tensor(Int(source))
    }
    
    public init(integerLiteral value: Float) {
        self.tensorPtr = torch.Tensor(value)
    }
    
    public var magnitude: Tensor {
        return Tensor(torch.abs(tensorPtr))
    }
    
    public typealias IntegerLiteralType = Float
    
    public typealias Magnitude = Tensor
    
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr + rhs.tensorPtr)!
    }
    
    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr - rhs.tensorPtr)!
    }
    
    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr * rhs.tensorPtr)!
    }
    
    public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr / rhs.tensorPtr)!
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
