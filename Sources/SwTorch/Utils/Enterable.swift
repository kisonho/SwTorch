//
//  Enterable.swift
//
//
//  Created by Kison Ho on 11/2/21.
//

import PythonKit

/// An enterable protocol that can be entered and exited
public protocol Enterable {
    /// Function to call when entering object
    mutating func enter()
    
    /// Function to call when exiting object
    mutating func exit()
}

/// disable gradients tracking
public struct NoGrad: Enterable {
    /// Previous status
    private var prev = Bool(torch.is_grad_enabled())!
    
    public func enter() {
        torch.set_grad_enabled(false)
    }
    
    public func exit() {
        torch.set_grad_enabled(prev)
    }
}

/// A Python with like statement
/// - Parameters:
///   - obj: PythonObject that accept with statement
///   - fn: function to call
public func with<T: Enterable>( _ obj: inout T, fn: (T) throws -> Any) throws -> Any {
    // enter
    obj.enter()
    defer { obj.exit() }
    return try fn(obj)
}
