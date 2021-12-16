//
//  Devices.swift
//  
//
//  Created by Kison Ho on 10/29/21.
//

import PythonKit

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

extension PySequential: DeviceMovable {
    public mutating func to(_ device: Device, id: Int?) {
        for m in modules {
            m.to(device, id: id)
        }
    }
}

extension WeightedModule {
    public mutating func to(_ device: Device, id: Int? = nil) {
        bias?.to(device, id: id)
        weight.to(device, id: id)
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
