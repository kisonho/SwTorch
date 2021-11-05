//
//  File.swift
//  
//
//  Created by Kison Ho on 11/4/21.
//

import PythonKit

/// Main metrics protocol
public protocol Metrics {
    /// Calculate method of metric
    /// - Parameters:
    ///   - yTrue: The label `Tensor`
    ///   - yPred: The target `Tensor`
    /// - Returns: A `Float` of current metric
    func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float
}

/// The metrics that calculates accuracy between two `Tensor`
public class Accuracy: Metrics {
    public func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        return Float(yTrue.equal(yPred).to(dtype: .float32).mean(axis: 0))!
    }
}

/// The mean absolute error between two `Tensor`
public class MAE: Metrics {
    public func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let diff = yTrue - yPred
        return Float(diff.abs().mean())!
    }
}

/// The mean squared error between two `Tensor`
public class MSE: Metrics {
    public func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let diff = (yTrue - yPred) ^ 2
        return Float(diff.abs().mean())!
    }
}

/// The python metrics mapping struct
public struct PyMetrics: Metrics {
    /// The pointer of python metrics
    public var metricsPtr: PythonObject
    
    public func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
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
    /// The axis of prediction
    var axis: Int = 1
    
    public override func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let y = yPred.argmax(axis: axis)
        return super.callAsFunction(yTrue: yTrue, yPred: y)
    }
}
