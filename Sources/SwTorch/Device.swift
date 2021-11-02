//
//  File.swift
//  
//
//  Created by Kison Ho on 10/29/21.
//

import PythonKit

/// Protocol for instances that can be move to a device
public protocol DeviceMovable {
    /// move current object to target device
    func to(_ device: Device)
}

extension DeviceMovable {
    /// Move current object to cpu
    func cpu() {
        self.to(.cpu)
    }
    
    /// Move current object to cuda
    func cuda() {
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
