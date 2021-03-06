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
    var lr: Double { get set }
    
    /// The optimizer where learning rate is
    var optimizer: OptimizerType { get set }
    
    /// Update the learning rate for each step
    /// - Returns: A `Double` of updated learning rate
    func updateLr() -> Double
}

public extension LrScheduler {
    /// Call for each step
    mutating func step() {
        lr = updateLr()
        self.optimizer.lr = lr
        self.currentStep += 1
    }
}

public struct ConstantLr<OptimizerType: Optimizer>: LrScheduler {
    public var currentStep: Int = 0
    
    public var lr: Double
    
    public var optimizer: OptimizerType
    
    /// Constructor
    /// - Parameters:
    ///   - optimizer: The optimizer to be updated
    ///   - lr: target learning rate
    init(_ optimizer: OptimizerType, lr: Double) {
        self.lr = lr
        self.optimizer = optimizer
    }
    
    public func updateLr() -> Double {
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
    var gamma: Double
    
    public var lr: Double
    
    public var optimizer: OptimizerType
    
    /// Constructor
    /// - Parameters:
    ///   - optimizer: An `OptimizerType` to be updated
    ///   - gamma: A `Double` of exponention value
    ///   - initialLr: A `Double` of initial learning rate
    public init(_ optimizer: OptimizerType, gamma: Double, initialLr: Double) {
        self.gamma = gamma
        self.lr = initialLr
        self.optimizer = optimizer
    }
    
    public func updateLr() -> Double {
        return lr * gamma
    }
}

public struct MultiStepLr<OptimizerType: Optimizer>: LrScheduler {
    public var currentStep: Int = 0
    
    /// The exponential gamma
    var gamma: Double
    
    public var lr: Double
    
    /// Steps to decay the learning rate
    var milestones: Array<Int>
    
    public var optimizer: OptimizerType
    /// Constructor
    /// - Parameters:
    ///   - optimizer: An `OptimizerType` to be updated
    ///   - gamma: A `Double` of exponention value
    ///   - initialLr: A `Double` of initial learning rate
    ///   - milestones: An `Array` of steps in `Int`
    public init(_ optimizer: OptimizerType, gamma: Double, initialLr: Double, milestones: Array<Int>) {
        self.optimizer = optimizer
        self.gamma = gamma
        self.lr = initialLr
        self.milestones = milestones
    }
    
    public func updateLr() -> Double {
        // decay gamma when step
        if milestones.contains(currentStep) {
            return lr * gamma
        } else {
            return lr
        }
    }
}

/// Main optimizer protocol
public protocol Optimizer {
    /// Parameter groups in an optimizer
    var lr: Double { get set }
    
    /// update parameters for one step
    func step()
    
    /// Clear optimizer gradient
    func zeroGrad(setToNone: Bool)
}

/// A python optimizer
public struct PyOptimizer: ConvertibleFromPython, Optimizer {
    public var lr: Double { get {
        return lrPointer
    } set(newLr) {
        lrPointer = newLr
        
        // loop for each group
        for g in optimizerPtr.param_groups {
            g["lr"] = PythonObject(newLr)
        }
    }}
    
    private var lrPointer: Double
    
    /// The pointer of torch.optim.Optimizer
    var optimizerPtr: PythonObject
    
    public init?(_ object: PythonObject) {
        self.optimizerPtr = object
        lrPointer = Double(optimizerPtr.param_groups[0]["lr"])!
    }
    
    public func step() {
        self.optimizerPtr.step()
    }
    
    public func zeroGrad(setToNone: Bool = false) {
        self.optimizerPtr.zero_grad(set_to_none: setToNone)
    }
}

extension PyOptimizer: PythonConvertible {
    public var pythonObject: PythonObject {
        return self.optimizerPtr
    }
}
