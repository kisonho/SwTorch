//
//  Parallel.swift
//  
//
//  Created by Kison Ho on 11/12/21.
//

import Foundation
import PythonKit

/// Data parallel module for multi-gpus support
public protocol DataParallelable {
    /// Target module type to parallel
    associatedtype DataParallelModuleType: Module & DeviceMovable
    
    /// The function to data parallel target module
    /// - Returns: A  data paralleled `ModuleType` which implements `Module`
    func dataParallel() -> DataParallelModuleType
}

/// A special module structure for multi-gpus support
public struct DataParalleledModule<M: DeviceMovable & Module>: Module {
    /// The indices of target devices
    let devices: Array<Int>
    
    /// The target module in `M`
    public var module: M
    
    /// The paralleled modules in devices
    var parallelModules: Array<M> = []
    
    public var parameters: Array<Tensor> { get {
        return module.parameters
    }}
    
    public var stateDict: [String : PythonObject?] { get {
        return module.stateDict
    } set {
        module.stateDict = newValue
    }}
    
    public init(_ file: URL) {
        self.module = M(file)
        devices = searchAllDevices()
    }
    
    public init(_ module: M, devices: [Int]? = nil) {
        self.module = module
        self.devices = devices ?? searchAllDevices()
    }
    
    public func copy() -> DataParalleledModule<M> {
        return DataParalleledModule(module.copy(), devices: devices)
    }
    
    public func eval() {
        self.module.eval()
    }
    
    public func forward(_ x: Tensor) -> Tensor {
        // initialize data parallel
        guard devices.count > 0 else { return module(x) }
        let batchPerGPU = Int(ceil(Double(x.shape[0] / devices.count)))
        let outputDevice = devices[0]
        var xInGPUs = Array<Tensor>()
        var yInGPUs = Array<Tensor>()
        
        // loop for each device
        for d in devices {
            let i = d * batchPerGPU
            let t = Array(x)[i ..< i + batchPerGPU]
            xInGPUs.append(Tensor(value: Array(t)))
        }
        
        // forward
        for (i, xInGPU) in xInGPUs.enumerated() {
            var y = parallelModules[i](xInGPU)
            y.to(.cuda, id: outputDevice)
            yInGPUs.append(y)
        }
        
        let y = Tensor(torch.nn.parallel.data_parallel.gather(yInGPUs, outputDevice, dim: 0))
        return y
    }
    
    public func toPyModule() -> PyModule {
        let m = module.toPyModule()
        return m.dataParallel()
    }
    
    public func train() {
        self.module.train()
    }
    
    public func save(_ file: URL) {
        self.module.save(file)
    }
}

extension DataParalleledModule: DeviceMovable {
    mutating public func to(_ device: Device, id: Int? = nil) {
        // initialize moving
        guard id == nil else { return }
        parallelModules.removeAll()
        
        // copy modules
        for d in devices {
            var m = module.copy()
            m.to(device, id: d)
            parallelModules.append(m)
        }
    }
}

extension PySequential: DataParallelable {
    public typealias DataParallelModuleType = PySequential
    
    public func dataParallel() -> PySequential {
        // initialize data paralleled modules
        var dataParalleledModules = Array<PyModule>()
        
        // data parallel each module
        for m in modules {
            dataParalleledModules.append(m.dataParallel())
        }
        
        // create data paralleled Sequential instance
        return PySequential(dataParalleledModules)
    }
}

extension WeightedModule where DataParallelModuleType == DataParalleledModule<Self> {
    public func dataParallel() -> DataParallelModuleType {
        return DataParalleledModule(self)
    }
}
