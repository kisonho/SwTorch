//
//  Devices.swift
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
    mutating func to(_ device: Device, id: Int?)
}

public extension DeviceMovable {
    /// Move current object to cpu
    mutating func cpu() {
        self.to(.cpu, id: nil)
    }
    
    /// Move current object to cuda
    mutating func cuda() {
        self.to(.cuda, id: nil)
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
    public func to(_ device: Device, id: Int? = nil) {
        let d = id == nil ? device.rawValue : "\(device.rawValue):\(id!)"
        self.modulePtr.to(torch.device(d))
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

extension PySequential: DeviceMovable {
    public mutating func to(_ device: Device, id: Int?) {
        for m in modules {
            m.to(device, id: id)
        }
    }
}

/// Function to search all available devices
/// - Returns: A list of divice indices
func searchAllDevices() -> Array<Int> {
    // check if cuda is available
    if Bool(torch.cuda.is_available()) == true {
        let numDevices = Int(torch.cuda.device_count())!
        return Array(0 ..< numDevices)
    } else {
        return []
    }
}
