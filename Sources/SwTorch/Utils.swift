//
//  File.swift
//  
//
//  Created by Kison Ho on 11/2/21.
//

import PythonKit

/// An enterable protocol that can be entered and exited
public protocol Enterable {
    /// Function to call when entering object
    func enter()
    
    /// Function to call when exiting object
    func exit()
}

extension PythonObject : Enterable {
    public func enter() {
        self.pythonObject.__enter__()
    }
    
    public func exit() {
        self.pythonObject.__exit__()
    }
}

/// A Python with like statement
/// - Parameters:
///   - obj: PythonObject that accept with statement
///   - fn: function to call
public func with<T: Enterable>(_ obj: T, fn: (T) throws -> Any) throws -> Any {
    // enter
    obj.enter()
    defer { obj.exit() }
    return try fn(obj)
}
