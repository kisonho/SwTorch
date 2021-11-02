//
//  File.swift
//
//
//  Created by Kison Ho on 10/26/21.
//

import PythonKit

/// import required modules
let np = Python.import("numpy")
public let torch = Python.import("torch")

/// Main Validation Protocol
public protocol EvaluatingManager {
    associatedtype ModuleType: Module
    
    var useMultiGPUs: Bool { get }
    var device: Device { get }
    var model: ModuleType { get set }
    
    /// calculate metrics
    /// - Returns: A `Dictionary` of metrics
    func calculateMetrics(yTrue: Tensor, yPred: Tensor) -> [String: Float]
    
    /// calculate loss
    /// - Returns: A `Tensor` of loss
    func calculateLoss(yTrue: Tensor, yPred: Tensor) -> Tensor
    
    /// On every batch ends
    func onBatchEnd(batch: Int, result: [String: Float])
}

public extension EvaluatingManager {
    /// Main validation function
    /// - Parameter dataLoader: A torch.utils.data.DataLoader to load the dataset
    /// - Returns: A Dictionary of validation results
    func validate(_ dataLoader: PythonObject) throws -> [String: Float] {
        // no gradients
        let valResultList = try with(torch.no_grad()) {
            // initialize validation
            var resultList: [String: Array<Float>] = [:]
            
            // batch loop
            for (batch, example) in Array(dataLoader).enumerated() {
                // extract example
                let (xTest, yTest) = example.tuple2
                
                // train step
                let result = valStep(Tensor(xTest), Tensor(yTest))
                onBatchEnd(batch: batch, result: result)
                
                // append to list
                for (key, m) in result {
                    if batch > 0 {
                        resultList[key]!.append(m)
                    } else {
                        resultList[key] = [m]
                    }
                }
            }
            
            return resultList
        }
    
        // initialize val result
        var valResult: [String: Float] = [:]
        
        // calculate mean
        for (key, m) in valResultList as! [String: Array<Float>] {
            valResult[key] = Float(torch.Tensor(m).mean())!
        }
        
        return valResult
    }
    
    /// Validation for a single step
    /// - Parameters:
    ///   - input: A torch.Tensor for input
    ///   - label: A torch.Tensor for label
    /// - Returns: A Dictionary of result of validation
    func valStep(_ xTest: Tensor, _ yTest: Tensor) -> [String: Float] {
        // move data to GPU
        if useMultiGPUs != true { xTest.to(device) }
        yTest.to(device)
        
        // forward pass
        let y = model(xTest)
        let loss = calculateLoss(yTrue: yTest, yPred: y)
        var metrics = calculateMetrics(yTrue: yTest, yPred: y)
        metrics["loss"] = Float(loss.mean())!
        
        // backward pass
        return metrics
    }
}

/// Main Training Protocol
public protocol TrainingManager: EvaluatingManager {
    associatedtype LrSchedulerType: LrScheduler
    associatedtype OptimizerType: Optimizer
    
    var lrScheduler: LrSchedulerType? { get set }
    var optimizer: OptimizerType { get }
    
    /// On every epoch starts
    func onEpochStart(epoch: Int, totalEpochs: Int)
    
    /// On every epoch ends
    /// - Returns: A Bool of flag if this result is best
    func onEpochEnd(epoch: Int, totalEpochs: Int, trainingResult: [String: Float], valResult: [String: Float]?) -> Bool
}

public extension TrainingManager {
    /// Backward pass
    /// - Parameters:
    ///   - loss: A torch.Tensor of loss
    func backward(_ loss: Tensor) {
        loss.backward()
        optimizer.step()
    }
    
    /// Main training function
    /// - Parameters:
    ///   - trainingDatasetLoader: A torch.utils.data.DataLoader to load the training dataset
    ///   - epochs: An Int of number of epochs
    ///   - initialEpoch: An Int of the starting epoch index
    ///   - validationDatasetLoader: An optional torch.utils.data.DataLoader to load validation dataset
    /// - Returns: An optional Dictionary of metrics (if validationDatasetLoader is not nil)
    mutating func train(trainingDatasetLoader: PythonObject, epochs: Int, initialEpoch: Int = 0, validationDatasetLoader: PythonObject? = nil) throws -> [String: Float]? {
        // initialize training
        var bestResult: [String: Float]? = nil
        
        // epoch loop
        for epoch in initialEpoch ..< epochs {
            // initialize epoch
            var resultList: [String: Array<Float>] = [:]
            onEpochStart(epoch: epoch, totalEpochs: epochs)
            
            // batch loop
            for (batch, example) in Array(trainingDatasetLoader).enumerated() {
                // extract example
                let xTrain = Tensor(example.tuple2.0)!
                let yTrain = Tensor(example.tuple2.1)!
                if useMultiGPUs != true { xTrain.to(device) }
                yTrain.to(device)
                
                // train step
                let result = trainStep(xTrain, yTrain)
                
                // append to list
                for (key, m) in result {
                    if batch > 0 {
                        resultList[key]!.append(m)
                    } else {
                        resultList[key] = [m]
                    }
                }
                
                // end batch callback
                onBatchEnd(batch: batch, result: result)
            }
            
            // initialize training result
            var trainingResult: [String: Float] = [:]
            var valResult: [String: Float]? = nil
            
            // calculate mean
            for (key, m) in resultList {
                trainingResult[key] = Float(torch.Tensor(m).mean())!
            }
            
            // validation
            valResult = validationDatasetLoader != nil ? try self.validate(validationDatasetLoader!) : nil
            
            // end epoch callback
            let isBest = self.onEpochEnd(epoch: epoch, totalEpochs: epochs, trainingResult: trainingResult, valResult: valResult)
            
            // set best result
            bestResult = isBest == true ? valResult : bestResult
            
            // step lr scheduler
            lrScheduler?.step()
        }
        
        return bestResult
    }
    
    /// Train for one step
    /// - Parameters:
    ///   - input: A torch.Tensor for input
    ///   - label: A torch.Tensor for label
    /// - Returns: A Dictionary of results
    func trainStep(_ xTrain: Tensor, _ yTrain: Tensor) -> [String: Float] {
        // forward pass
        optimizer.zeroGrad()
        let y = model(xTrain)
        let loss = calculateLoss(yTrue: yTrain, yPred: y)
        var metrics = calculateMetrics(yTrue: yTrain, yPred: y)
        metrics["loss"] = Float(loss.mean())!
        
        // backward pass
        backward(loss)
        return metrics
    }
}

/// With like statement for python objects
/// - Parameters:
///   - obj: PythonObject that accept with statement
///   - fn: function to call
public func with(_ obj: PythonObject, fn: () throws -> Any) throws -> Any {
    // enter
    obj.__enter__()
    defer { obj.__exit__() }
    return try fn()
}
