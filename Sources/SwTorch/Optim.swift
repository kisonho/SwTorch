//
//  File.swift
//  
//
//  Created by Kison Ho on 10/29/21.
//

import PythonKit

/// The protocol to update learning rate for each step
public protocol LrScheduler {
    /// The optimizer type
    associatedtype OptimizerType: Optimizer
    
    /// Current learning rate
    var lr: Float { get set }
    
    /// The optimizer where learning rate is
    var optimizer: OptimizerType { get set }
    
    /// Update the learning rate for each step
    /// - Returns: A `Float` of updated learning rate
    func updateLr() -> Float
}

extension LrScheduler {
    /// Call for each step
    mutating func step() {
        // get updated lr
        lr = updateLr()
        
        // update lr in optimizer
        for (i, _) in optimizer.paramGroups.enumerated() {
            optimizer.paramGroups[i]["lr"] = lr
        }
    }
}

/// Exponention learning rate scheduler
public struct ExponentionLr<OptimizerType: Optimizer>: LrScheduler {
    /// The exponential gamma
    var gamma: Float
    
    /// Current learning rate
    public var lr: Float
    
    /// The optimizer where learning rate is
    public var optimizer: OptimizerType
    
    /// Constructor
    /// - Parameters:
    ///   - optimizer: An `OptimizerType` to be updated
    ///   - gamma: A `Float` of exponention value
    ///   - initialLr: A `Float` of initial learning rate
    init(_ optimizer: OptimizerType, gamma: Float, initialLr: Float) {
        self.gamma = gamma
        self.lr = initialLr
        self.optimizer = optimizer
    }
    
    /// Update the learning rate for each step
    /// - Returns: A `Float` of updated learning rate
    public func updateLr() -> Float {
        return lr * gamma
    }
}

/// Main optimizer protocol
public protocol Optimizer {
    /// Parameter groups in an optimizer
    var paramGroups: Array<[String: Float]> { get set }
    
    /// update parameters for one step
    func step()
    
    /// Clear optimizer gradient
    func zeroGrad()
}

/// A python optimizer
public struct PyOptimizer: ConvertibleFromPython, Optimizer {
    /// Parameter groups in an optimizer
    public var paramGroups: Array<[String : Float]> { get {
        return Array(optimizerPtr.param_groups)!
    } set(newGroups) {
        // loop for each group
        for (i, group) in newGroups.enumerated() {
            for (key, value) in group {
                optimizerPtr.param_groups[i][key] = PythonObject(value)
            }
        }
    }}
    
    var optimizerPtr: PythonObject
    
    public init?(_ object: PythonObject) {
        self.optimizerPtr = object
    }
    
    /// Performs a single optimization step (parameter update)
    public func step() {
        self.optimizerPtr.step()
    }
    
    /// performs a
    public func zeroGrad() {
        self.optimizerPtr.zero_grad()
    }
}

extension PyOptimizer: PythonConvertible {
    public var pythonObject: PythonObject {
        return self.optimizerPtr
    }
}
