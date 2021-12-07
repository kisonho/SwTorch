//
//  Metrics.swift
//  
//
//  Created by Kison Ho on 11/4/21.
//

import PythonKit

// import required python modules
fileprivate let F = Python.import("torch.nn.functional")

/// The metrics that calculate BCE with logits loss between two `Tensor`
public class BCELoss: Loss {
    /// A manual rescaling weight
    var weight: Tensor? = nil
    
    public init() {}
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Tensor {
        return Tensor(F.binary_cross_entropy(yPred, yTrue, weight))
    }
}

/// The metrics that calculate BCE with logits loss between two `Tensor`
public class BCEWithLogitsLoss: Loss {
    /// A manual rescaling weight
    var weight: Tensor? = nil
    
    public init() {}
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Tensor {
        return Tensor(F.binary_cross_entropy_with_logits(yPred, yTrue, weight))
    }
}

/// The metrics that calculate cross entropy loss between two `Tensor`
public class CrossEntropyLoss: Loss {
    /// A manual rescaling weight
    var weight: Tensor? = nil
    
    /// The target value that is ignored and does not contribute to the input gradient
    var ignoreIndex: Int = -100
    
    public init() {}
    
    /// Constructor
    /// - Parameters:
    ///   - weight: A `Tensor` of loss weight
    ///   - ignoreIndex: An `Int` of target value that is ignored and does not contribute to the input gradient
    public init(weight: Tensor, ignoreIndex: Int = -100) {
        self.weight = weight
        self.ignoreIndex = ignoreIndex
    }
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Tensor {
        return Tensor(F.cross_entropy(yPred, yTrue, weight: weight, ignore_index: ignoreIndex))
    }
}

/// The metrics that calculate the Kullback-Leibler divergence loss
public class KLDivLoss: Loss {
    /// A `Bool` flag of if passing the log space
    var logTarget = false
    
    /// A `Bool` flag of if averaging by minibatch
    var reduce = true
    
    /// Constructor
    /// - Parameters:
    ///   - reduce: A `Bool` flag of if averaging by minibatch
    ///   - logTarget: A `Bool` flag of if passing the log space
    public init(reduce: Bool = true, logTarget: Bool = false) {
        self.logTarget = logTarget
        self.reduce = reduce
    }
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Tensor {
        let reduction = reduce == true ? "batchmean" : "none"
        return Tensor(F.kl_div(yPred, yTrue, reduction: reduction, log_target: logTarget))
    }
}

public protocol Loss {
    /// Calculate method of metric
    /// - Parameters:
    ///   - yTrue: The target `Tensor`
    ///   - yPred: The input `Tensor`
    /// - Returns: A `Tensor` of current metric
    func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Tensor
}

extension Loss {
    /// Calculate method of metric
    /// - Parameters:
    ///   - yTrue: The target `Tensor`
    ///   - yPred: The input `Tensor`
    /// - Returns: A `Tensor` of current metric
    @available(*, deprecated, message: "Use without labels for both parameters: callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Tensor")
    func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Tensor {
        return callAsFunction(yTrue, yPred)
    }
}

/// Main metrics protocol
public protocol Metrics {
    /// Calculate method of metric
    /// - Parameters:
    ///   - yTrue: The target `Tensor`
    ///   - yPred: The input `Tensor`
    /// - Returns: A `Float` of current metric
    func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Float
}

extension Metrics {
    /// Calculate method of metric
    /// - Parameters:
    ///   - yTrue: The target `Tensor`
    ///   - yPred: The input `Tensor`
    /// - Returns: A `Tensor` of current metric
    @available(*, deprecated, message: "Use without labels for both parameters: callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Tensor")
    func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        return callAsFunction(yTrue, yPred)
    }
}

/// The metrics that calculates accuracy between two `Tensor`
public class Accuracy: Metrics {
    public init() {}
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Float {
        return Float(yTrue.equal(yPred).to(dtype: .float32).mean())!
    }
}

/// The mean absolute error between two `Tensor`
public class MAE: Metrics {
    public init() {}
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Float {
        let diff = yTrue - yPred
        return Float(diff.abs().mean())!
    }
}

/// The mean squared error between two `Tensor`
public class MSE: Metrics {
    public init() {}
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Float {
        let diff = (yTrue - yPred) ^ 2
        return Float(diff.abs().mean())!
    }
}

/// The python metrics mapping struct
public struct PyMetrics: Metrics {
    /// The pointer of python metrics
    public var metricsPtr: PythonObject
    
    public func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Float {
        return Float(metricsPtr(yPred, yTrue))!
    }
}

extension PyMetrics: ConvertibleFromPython {
    public init(_ object: PythonObject) {
        self.metricsPtr = object
    }
}

/// The metrics that calculate accuracy between a real label `Tensor` and logits `Tensor`
public final class SparseCategoricalAccuracy: Accuracy {
    /// The dimention of prediction
    var dim: Int = 1
    
    public override func callAsFunction(_ yTrue: Tensor, _ yPred: Tensor) -> Float {
        let y = yPred.argmax(dim: dim)
        return super.callAsFunction(yTrue, y)
    }
}
