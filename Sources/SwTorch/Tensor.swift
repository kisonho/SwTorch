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
    public var pythonObject: PythonObject
    
    /// Constructor
    /// - Parameters:
    ///   - value: A PythonConvertible real value of the tensor
    ///   - shape:
    public init(_ value: PythonConvertible) {
        self.pythonObject = torch.Tensor(value)
    }
}

extension Tensor: ConvertibleFromPython {
    /// Constructor
    /// - Parameters:
    ///   - object: The python object that convert from
    public init?(_ object: PythonObject) {
        pythonObject = object
    }
    
    /// Get the max index in axis
    /// - Parameter axis: Int of axis of argmax
    /// - Returns: A Tensor of argmax of current tensor
    public func argmax(axis: Int? = nil) -> Tensor {
        return Tensor(pythonObject.argmax(axis: axis))
    }
}

extension Tensor: DeviceMovable {
    /// Move current module to a target device
    /// - Parameter device: A String of device
    public func to(_ device: Device) {
        self.pythonObject.to(torch.device(device.rawValue))
    }
}
