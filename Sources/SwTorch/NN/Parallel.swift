//
//  Parallel.swift
//  
//
//  Created by Kison Ho on 11/12/21.
//

import Foundation
import PythonKit

extension Conv2D: DataParallelable {
    typealias DataParallelModuleType = DataParalleledModule<Conv2D>
    
    func dataParallel() -> DataParalleledModule<Conv2D> {
        return DataParalleledModule(self)
    }
}

/// A special module structure for multi-gpus support
struct DataParalleledModule<M: DeviceMovable & Module>: Module {
    /// The indices of target devices
    let devices: Array<Int>
    
    /// The target module in `M`
    public var module: M
    
    /// The paralleled modules in devices
    var parallelModules: Array<M> = []
    
    var parameters: Array<Tensor> { get {
        return module.parameters
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
    
    public mutating func loadStateDict(_ dict: [String : PythonObject?]) {
        self.module.loadStateDict(dict)
    }
    
    func toPyModule() -> PyModule {
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

extension Linear: DataParallelable {
    typealias DataParallelModuleType = DataParalleledModule<Linear>
    
    func dataParallel() -> DataParalleledModule<Linear> {
        return DataParalleledModule(self)
    }
}
