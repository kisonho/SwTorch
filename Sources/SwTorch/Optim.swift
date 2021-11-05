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

public extension LrScheduler {
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

public struct ConstantLr<OptimizerType: Optimizer>: LrScheduler {
    public var lr: Float
    
    public var optimizer: OptimizerType
    
    /// Constructor
    /// - Parameters:
    ///   - optimizer: The optimizer to be updated
    ///   - lr: target learning rate
    init(_ optimizer: OptimizerType, lr: Float) {
        self.lr = lr
        self.optimizer = optimizer
    }
    
    public func updateLr() -> Float {
        return lr
    }
    
    mutating func step() {
        return
    }
}

/// Exponention learning rate scheduler
public struct ExponentionLr<OptimizerType: Optimizer>: LrScheduler {
    /// The exponential gamma
    var gamma: Float
    
    public var lr: Float
    
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
    
    /// The pointer of torch.optim.Optimizer
    var optimizerPtr: PythonObject
    
    public init?(_ object: PythonObject) {
        self.optimizerPtr = object
    }
    
    public func step() {
        self.optimizerPtr.step()
    }
    
    public func zeroGrad() {
        self.optimizerPtr.zero_grad()
    }
}

extension PyOptimizer: PythonConvertible {
    public var pythonObject: PythonObject {
        return self.optimizerPtr
    }
}
