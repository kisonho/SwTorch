//
//  File.swift
//  
//
//  Created by Kison Ho on 10/28/21.
//

import PythonKit

/// main tensor struct
public struct Tensor: PythonConvertible {
    /// The torch.Tensor python object
    var tensorPtr: PythonObject
    public var pythonObject: PythonObject { get {
        return self.tensorPtr
    }}
    
    /// Constructor
    /// - Parameters:
    ///   - value: A PythonConvertible real value of the tensor
    ///   - shape:
    public init(_ value: PythonConvertible) {
        self.tensorPtr = torch.Tensor(value)
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
        return Tensor(pythonObject.argmax(axis: axis))!
    }
    
    /// calculate the mean
    /// - Returns: A `Tensor` of mean of current tensor
    public func mean() -> Tensor {
        return Tensor(self.tensorPtr.mean())!
    }
}

extension Tensor : CustomStringConvertible {
    /// A textual description of this `Tensor`
    public var description: String {
        return String(tensorPtr)!
    }
}

extension Tensor: DeviceMovable {
    /// Move current module to a target device
    /// - Parameter device: A `String` of device
    public func to(_ device: Device) {
        self.pythonObject.to(torch.device(device.rawValue))
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

public extension Tensor {
    static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr + rhs.tensorPtr)!
    }
    
    static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr - rhs.tensorPtr)!
    }
    
    static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr * rhs.tensorPtr)!
    }
    
    static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        return Tensor(lhs.tensorPtr / rhs.tensorPtr)!
    }
    
    static func += (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs + rhs
    }
    
    static func -= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs - rhs
    }
    
    static func *= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs * rhs
    }
    
    static func /= (lhs: inout Tensor, rhs: Tensor) {
        lhs = lhs / rhs
    }
}

extension Double {
    public init?(_ tensor: Tensor) {
        self = Double(tensor.tensorPtr)!
    }
}

extension Float {
    public init?(_ tensor: Tensor) {
        guard let v = Float(tensor.tensorPtr) else { return nil }
        self.init(v)
    }
}

extension Int {
    public init?(_ tensor: Tensor) {
        guard let v = Int(tensor.tensorPtr) else { return nil }
        self.init(v)
    }
}
