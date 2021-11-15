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
    
    /// Set target module into training mode
    func train()
    
    /// Save current model
    /// - Parameter file: A `String` of file path to be saved
    func save(_ file: URL)
}

public extension Module {
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
            return Array(modulePtr.modules())!
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
        self.modulePtr = torch.load(file.absoluteString)
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
    
    public func train() {
        modulePtr.train()
    }
    
    public func save(_ file: URL) {
        // import required python library
        let os = Python.import("os")
        
        // get directory from URL
        let dir = file.deletingLastPathComponent()
        
        // create directory
        os.makedirs(dir.absoluteString, exist_ok: true)
        
        // save module
        torch.save(modulePtr, file.absoluteString)
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
}

/// A sequential module of `PyModules`
public struct PySequential: Module {
    /// The modules list
    var modules: Array<PyModule>
    
    var parameters: Array<Tensor> { get {
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
    public init(_ modules: Array<PyModule>) {
        self.modules = modules
    }
    
    public init(_ file: URL) {
        // list directories in file URL
        let os = Python.import("os")
        let dirs = Array<String>(os.listdir(file.absoluteString))!
        
        // initialize loading modules
        modules = []
        
        // loop for each dir
        for dir in dirs {
            if dir.prefix(1) != "." {
                let loadedModule = PyModule(URL(fileURLWithPath: dir))
                modules.append(loadedModule)
            }
        }
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
        // initialize sequential pointer
        let pySequential = torch.nn.Sequential()
        
        // add modules
        for (i, m) in modules.enumerated() {
            pySequential.add_module(i, m)
        }
        
        // save to file
        torch.save(pySequential, file.description)
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
