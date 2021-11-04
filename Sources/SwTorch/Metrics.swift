//
//  File.swift
//  
//
//  Created by Kison Ho on 11/4/21.
//

import Foundation

/// Main metrics protocol
public protocol Metrics {
    /// Basic score
    var score: Array<Float> { get set }
    
    /// Calculate method of metric
    /// - Returns: A `Tensor` of current metric
    func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float
}

extension Metrics {
    /// The final result of metrics
    var result: Float { get {
        return Float(Tensor(score, dtype: .float32).mean())!
    }}
    
    /// The main update method to calculate metric and update score
    /// - Parameters:
    ///   - yTrue: A `Tensor` of label
    ///   - yPred: A `Tensor` of module output
    /// - Returns: A `Float` of current metric
    mutating func update(yTrue: Tensor, yPred: Tensor) -> Float {
        let metric = callAsFunction(yTrue: yTrue, yPred: yPred)
        score.append(metric)
        return metric
    }
}

/// The metrics that calculates accuracy between two `Tensor`
public class Accuracy: Metrics {
    public var score: Array<Float> = []
    
    public init() {}
    
    public func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        return Float(yTrue.equal(yPred).to(dtype: .float32).mean(axis: 0))!
    }
}

/// The mean absolute error between two `Tensor`
public class MAE: Metrics {
    public var score: Array<Float> = []
    
    public init() {}
    
    public func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let diff = yTrue - yPred
        return Float(diff.abs().mean())!
    }
}

/// The mean squared error between two `Tensor`
public class MSE: Metrics {
    public var score: Array<Float> = []
    
    public init() {}
    
    public func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let diff = (yTrue - yPred) ^ 2
        return Float(diff.abs().mean())!
    }
}

/// The metrics that calculate accuracy between a real label `Tensor` and logits `Tensor`
public final class SparseCategoricalAccuracy: Accuracy {
    public override func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let y = yPred.argmax(axis: 1)
        return super.callAsFunction(yTrue: yTrue, yPred: y)
    }
}
