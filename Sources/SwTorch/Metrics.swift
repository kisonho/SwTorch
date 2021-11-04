//
//  File.swift
//  
//
//  Created by Kison Ho on 11/4/21.
//

import Foundation

/// Main metrics protocol
protocol Metrics {
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
class Accuracy: Metrics {
    var score: Array<Float> = []
    
    func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        return Float(yTrue.equal(yPred).to(dtype: .float32).mean())!
    }
}

/// The mean absolute error between two `Tensor`
class MAE: Metrics {
    var score: Array<Float> = []
    
    func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let diff = yTrue - yPred
        return Float(diff.abs().mean())!
    }
}

/// The mean squared error between two `Tensor`
class MSE: Metrics {
    var score: Array<Float> = []
    
    func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let diff = (yTrue - yPred) ^ 2
        return Float(diff.abs().mean())!
    }
}

/// The metrics that calculate accuracy between a real label `Tensor` and logits `Tensor`
final class SparseCategoricalAccuracy: Accuracy {
    override func callAsFunction(yTrue: Tensor, yPred: Tensor) -> Float {
        let y = yPred.argmax(axis: 1)
        return super.callAsFunction(yTrue: y, yPred: yPred)
    }
}
