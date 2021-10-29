//
//  File.swift
//  
//
//  Created by Kison Ho on 10/29/21.
//

import PythonKit

/// Protocol for instances that can be move to a device
public protocol DeviceMovable {
    func to(_ device: Device)
}

public enum Device: String {
    case cpu = "cpu"
    case cuda = "cuda"
}
