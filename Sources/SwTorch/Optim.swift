//
//  File.swift
//  
//
//  Created by Kison Ho on 10/29/21.
//

import PythonKit

/// Main optimizer protocol
public protocol Optimizer: PythonConvertible {
    var optimizerPtr: PythonObject { get }
}

extension Optimizer {
    /// Performs a single optimization step (parameter update)
    func step() {
        self.optimizerPtr.step()
    }
    
    /// performs a
    func zero_grad() {
        self.optimizerPtr.zero_grad()
    }
}

/// Standard SGD optimizer
public struct SGD: Optimizer {
    public var optimizerPtr: PythonObject
    public var pythonObject: PythonObject { get {
        return optimizerPtr
    }}
    
    /// Constructor
    /// - Parameters:
    ///   - params: An `Array<Tensor>` of parameters to be updated
    ///   - lr: A `Float` of learning rate
    ///   - momentum: A `Float` of momentum
    ///   - dampening: A `Float` of dampening for momentum
    ///   - weightDecay: A `Float` of L2 penalty
    ///   - nesterov: A `Bool` flag to enable Nesterov Momentum
    init(_ params: Array<Tensor>, lr: Float, momentum: Float = 0, dampening: Float = 0, weightDecay: Float = 0, nesterov: Bool = false) {
        self.optimizerPtr = torch.optim.SGD(params, lr: lr, momentum: momentum, dampening: dampening, weight_decay: weightDecay, nesterov: nesterov)
    }
}
