//
//  Module.swift
//  Torch
//
//  Created by Kison Ho on 10/28/21.
//

import PythonKit
import Foundation

/// Main module protocol
public protocol Module: DeviceMovable {
    /// Main forward function
    /// - Parameter x: A `Tensor` of input
    /// - Returns: A `Tensor` of output
    func forward(_ x: Tensor) -> Tensor
    
    /// Load state dict from saved module
    /// - Parameter file: A String of saved module path
    func loadStateDict<DictValueType: PythonConvertible>(_ dict: [String: DictValueType])
    
    /// Save current model
    /// - Parameter file: A `String` of file path to be saved
    func save(_ file: URL)
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
    
    /// Main forward pass function
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
    
    public init?(_ object: PythonObject) {
        modulePtr = object
    }
    
    public func forward(_ x: Tensor) -> Tensor {
        return Tensor(modulePtr(x))
    }
    
    public func loadStateDict<DictValueType: PythonConvertible>(_ dict: [String: DictValueType]) {
        self.modulePtr.loadStateDict(dict)
    }
    
    public func save(_ file: URL) {
        torch.save(modulePtr, file.absoluteString)
    }
    
    public func to(_ device: Device) {
        self.modulePtr.to(torch.device(device.rawValue))
    }
}

extension PyModule: PythonConvertible {
    public var pythonObject: PythonObject {
        return modulePtr
    }
}

public struct Sequential: Module {
    /// The modules list
    var modules: Array<Module> = []
    
    public func forward(_ x: Tensor) -> Tensor {
        // initialize input
        var x = x
        
        // loop for modules
        for m in modules {
            x = m(x)
        }
        
        return x
    }
    
    public func loadStateDict<DictValueType>(_ dict: [String : DictValueType]) where DictValueType : PythonConvertible {
        // loop for modules
        for (i, m) in modules.enumerated() {
            m.loadStateDict(dict[String(i)] as! [String: DictValueType])
        }
    }
    
    public func save(_ file: URL) {
        // loop for modules
        for (i, module) in modules.enumerated() {
            var moduleFile = file
            moduleFile.appendPathComponent(String(i))
            module.save(moduleFile)
        }
    }
    
    public mutating func to(_ device: Device) {
        // loop for modules
        for (i, _) in modules.enumerated() {
            modules[i].to(device)
        }
    }
}


