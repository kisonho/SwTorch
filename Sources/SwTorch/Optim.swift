//
//  Optim.swift
//  
//
//  Created by Kison Ho on 10/29/21.
//

import PythonKit

/// The protocol to update learning rate for each step
public protocol LrScheduler {
    /// The optimizer type
    associatedtype OptimizerType: Optimizer
    
    /// The current step index
    var currentStep: Int { get set }
    
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
        self.optimizer.lr = updateLr()
        self.currentStep += 1
    }
}

public struct ConstantLr<OptimizerType: Optimizer>: LrScheduler {
    public var currentStep: Int = 0
    
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
    public var currentStep: Int = 0
    
    /// The exponential gamma
    var gamma: Float
    
    public var lr: Float
    
    public var optimizer: OptimizerType
    
    /// Constructor
    /// - Parameters:
    ///   - optimizer: An `OptimizerType` to be updated
    ///   - gamma: A `Float` of exponention value
    ///   - initialLr: A `Float` of initial learning rate
    public init(_ optimizer: OptimizerType, gamma: Float, initialLr: Float) {
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
    var lr: Float { get set }
    
    /// update parameters for one step
    func step()
    
    /// Clear optimizer gradient
    func zeroGrad()
}

/// A python optimizer
public struct PyOptimizer: ConvertibleFromPython, Optimizer {
    public var lr: Float { get {
        return lrPointer
    } set(newLr) {
        lrPointer = newLr
        
        // loop for each group
        for g in optimizerPtr.param_groups {
            g["lr"] = PythonObject(lr)
        }
    }}
    
    private var lrPointer: Float
    
    /// The pointer of torch.optim.Optimizer
    var optimizerPtr: PythonObject
    
    public init?(_ object: PythonObject) {
        self.optimizerPtr = object
        lrPointer = Float(optimizerPtr.param_groups[0]["lr"])!
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
