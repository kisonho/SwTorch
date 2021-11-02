//
//  Module.swift
//  Torch
//
//  Created by Kison Ho on 10/28/21.
//

import PythonKit

/// Main module protocol
public protocol Module: DeviceMovable {
    /// Main forward function
    /// - Returns: A Tensor of output
    func forward(_ x: Tensor) -> Tensor
    
    /// Load saved state dict to target module
    func loadStateDict<DictValueType: PythonConvertible>(_ dict: [String: DictValueType])
    
    /// Save current model
    func save(_ file: String)
}

extension Module {
    /// Automatically get all parameters
    var parameters: Array<Tensor> { get {
        // initialize mirror
        let mirror = Mirror(reflecting: self)
        var params = Array<Tensor>()
        
        // loop for attributes
        for attr in mirror.children {
            if let param = attr.value as? Tensor {
                params.append(param)
            }
        }
        
        return params
    } }
    
    /// Main call function
    /// - Parameter x: A Tensor of input
    /// - Returns: A Tensor of output
    public func callAsFunction(_ x: Tensor) -> Tensor {
        return forward(x)
    }
}

/// Main PyTorch Module
public struct PyModule: ConvertibleFromPython, Module {
    /// The torch.nn.Module python object
    var modulePtr: PythonObject
    
    public var parameters: Array<Tensor> { get {
        // initialize getting parameters
        let pyParams = self.modulePtr.parameters()
        var params = Array<Tensor>()
        
        // loop for each param
        for pyParam in pyParams {
            params.append(Tensor(pyParam))
        }
        
        return params
    }}
    
    /// Constructor
    /// - Parameters:
    ///   - object: The python object in `torch.nn.Module` that convert from
    public init?(_ object: PythonObject) {
        modulePtr = object
    }
    
    /// Main forward function that forward the target torch.nn.Module
    /// - Parameter x: A Tensor of input
    /// - Returns: A Tensor of output
    public func forward(_ x: Tensor) -> Tensor {
        return Tensor(modulePtr(x))!
    }
    
    /// Load state dict from saved module
    /// - Parameter file: A String of saved module path
    public func loadStateDict<DictValueType: PythonConvertible>(_ dict: [String: DictValueType]) {
        self.modulePtr.loadStateDict(dict)
    }
    
    /// Move current module to a target device
    /// - Parameter device: A String of device
    public func to(_ device: Device) {
        self.modulePtr.to(torch.device(device.rawValue))
    }
    
    /// Save current python module to a file
    /// - Parameter file: A String of file location
    public func save(_ file: String) {
        torch.save(self.modulePtr, file)
    }
}

extension PyModule: PythonConvertible {
    public var pythonObject: PythonObject {
        return modulePtr
    }
}
