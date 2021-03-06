//
//  Modules.swift
//
//
//  Created by Kison Ho on 10/28/21.
//

import PythonKit
import Foundation

/// Main module protocol
public protocol Module {
    /// Create new instance by loading from file
    /// - Parameter file: A `String` of file path to be loaded
    init(_ file: URL)
    
    /// Deep copy a current module
    /// - Returns: The same class of current module
    func copy() -> Self
    
    /// Set target module into validation mode
    func eval()
    
    /// Main forward function
    /// - Parameter x: A `Tensor` of input
    /// - Returns: A `Tensor` of output
    func forward(_ x: Tensor) -> Tensor
    
    /// Load state dict from saved module
    /// - Parameter dict: A `Dictionary` of parameter name and optional value in `PythonObject`
    mutating func loadStateDict(_ dict: [String: PythonObject?])
    
    /// Method to convert current module to PyModule
    func toPyModule() -> PyModule
    
    /// Set target module into training mode
    func train()
    
    /// Save current model
    /// - Parameter file: A `String` of file path to be saved
    func save(_ file: URL)
}

public extension Module {
    /// Automatically get all modules as PyModule in current module
    var modules: Array<PyModule> { get {
        // initialize mirror
        let mirror = Mirror(reflecting: self)
        var pyModules = Array<PyModule>()
        
        // loop for attributes
        for attr in mirror.children {
            if let m = attr.value as? Module {
                pyModules.append(m.toPyModule())
            }
        }
        
        return pyModules
    }}
    
    /// Automatically get all parameters in current module
    var parameters: Array<Tensor> { get {
        // initialize mirror
        let mirror = Mirror(reflecting: self)
        var params = Array<Tensor>()
        
        // loop for attributes
        for attr in mirror.children {
            if let param = attr.value as? Tensor {
                params.append(param)
            } else if let m = attr.value as? Module {
                params += m.parameters
            }
        }
        
        return params
    } }
    
    /// Main forward pass function
    /// - Parameter x: A `Tensor` of input
    /// - Returns: A `Tensor` of output
    func callAsFunction(_ x: Tensor) -> Tensor {
        return forward(x)
    }
}

/// Main PyTorch Module
public struct PyModule: Module {
    /// The torch.nn.Module python object
    var modulePtr: PythonObject
    
    /// The sub `PyModule` inside this module
    var modules: Array<PyModule> {
        get {
            return Array(modulePtr.children())!
        }
    }
    
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
    
    public init(_ file: URL) {
        self.modulePtr = torch.load(file.absoluteString.replacingOccurrences(of: "file://", with: ""))
    }
    
    public func copy() -> PyModule {
        let copy = Python.import("copy")
        return PyModule(copy.deepcopy(modulePtr))!
    }
    
    public func eval() {
        modulePtr.eval()
    }
    
    public func forward(_ x: Tensor) -> Tensor {
        return Tensor(modulePtr(x))
    }
    
    public func loadStateDict(_ dict: [String: PythonObject?]) {
        self.modulePtr.loadStateDict(dict)
    }
    
    public func toPyModule() -> PyModule {
        return self
    }
    
    public func train() {
        modulePtr.train()
    }
    
    public func save(_ file: URL) {
        // import required python library
        let os = Python.import("os")
        
        // get directory from URL
        let dir = file.deletingLastPathComponent()
        
        // create directory
        os.makedirs(dir.absoluteString.replacingOccurrences(of: "file://", with: ""), exist_ok: true)
        
        // save module
        torch.save(modulePtr, file.absoluteString.replacingOccurrences(of: "file://", with: ""))
    }
}

extension PyModule: ConvertibleFromPython {
    public init?(_ object: PythonObject) {
        modulePtr = object
    }
}

extension PyModule: PythonConvertible {
    public var pythonObject: PythonObject {
        return modulePtr
    }
    
    /// Constructor to convert a `PySequential` to `PyModule`
    /// - Parameter sequential: A `PySequential` model to convert
    public init(sequential: PySequential) {
        // initialize sequential
        let seq = torch.nn.Sequential()
        
        // loop for modules
        for (i, m) in sequential.modules.enumerated() {
            seq.add_module(String(i), m)
        }
        
        // set python module pointer
        modulePtr = seq
    }
}

/// A sequential module of `PyModules`
public struct PySequential: Module {
    /// The modules list
    public var modules: Array<PyModule>
    
    public var parameters: Array<Tensor> { get {
        // initialize getting parameters
        var params = Array<Tensor>()
        
        // loop for each module
        for m in modules {
            params += m.parameters
        }
        
        return params
    }}
    
    /// Constructor
    /// - Parameter modules: An `Array<ModuleType>` of the modules
    public init(_ modules: Array<PyModule> = []) {
        self.modules = modules
    }
    
    public init(_ file: URL) {
        self.modules = PyModule(file).modules
    }
    
    public func copy() -> PySequential {
        // initialize copy
        var newModules = Array<PyModule>()
        
        // loop for each module
        for m in modules {
            newModules.append(m.copy())
        }
        
        return PySequential(newModules)
    }
    
    public func eval() {
        for m in modules {
            m.eval()
        }
    }
    
    public func forward(_ x: Tensor) -> Tensor {
        // initialize input
        var x = x
        
        // loop for modules
        for m in modules {
            x = m(x)
        }
        
        return x
    }
    
    public mutating func loadStateDict(_ dict: [String : PythonObject?]) {
        let pySequential = PyModule(torch.nn.Sequential(modules))!
        pySequential.loadStateDict(dict)
        modules = pySequential.modules
    }
    
    public func save(_ file: URL) {
        // save to file
        self.toPyModule().save(file)
    }
    
    public func toPyModule() -> PyModule {
        return PyModule(sequential: self)
    }
    
    public func train() {
        for m in modules {
            m.train()
        }
    }
    
    public mutating func to(_ device: Device) {
        for (i, _) in modules.enumerated() {
            modules[i].to(device)
        }
    }
}
