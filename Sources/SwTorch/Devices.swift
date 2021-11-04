//
//  File.swift
//  
//
//  Created by Kison Ho on 10/29/21.
//

import PythonKit

/// Data parallel module for multi-gpus support
public protocol DataParallelable {
    /// Target module type to parallel
    associatedtype DataParallelModuleType: Module
    
    /// The function to data parallel target module
    /// - Returns: A  data paralleled `ModuleType` which implements `Module`
    func dataParallel() -> DataParallelModuleType
}

/// Protocol for instances that can be move to a device
public protocol DeviceMovable {
    /// move current object to target device
    mutating func to(_ device: Device)
}

public extension DeviceMovable {
    /// Move current object to cpu
    mutating func cpu() {
        self.to(.cpu)
    }
    
    /// Move current object to cuda
    mutating func cuda() {
        self.to(.cuda)
    }
}

/// Available devices
public enum Device: String {
    /// cpu
    case cpu = "cpu"
    
    /// gpu
    case cuda = "cuda"
}

extension PyModule: DataParallelable {
    public typealias DataParallelModuleType = PyModule
    
    public func dataParallel() -> DataParallelModuleType {
        return PyModule(torch.nn.DataParallel(self.modulePtr))!
    }
}

extension PyModule: DeviceMovable {
    public func to(_ device: Device) {
        self.modulePtr.to(torch.device(device.rawValue))
    }
}

extension Sequential: DataParallelable {
    public typealias DataParallelModuleType = Sequential
    
    public func dataParallel() -> Sequential {
        // initialize data paralleled modules
        var dataParalleledModules = Array<PyModule>()
        
        // data parallel each module
        for m in modules {
            dataParalleledModules.append(m.dataParallel())
        }
        
        // create data paralleled Sequential instance
        return Sequential(dataParalleledModules)
    }
}
