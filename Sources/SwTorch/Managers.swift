//
//  File.swift
//
//
//  Created by Kison Ho on 10/26/21.
//

import PythonKit

/// import required modules
public let torch = Python.import("torch")

/// Main Validation Protocol
public protocol EvaluatingManager {
    /// The type of model
    associatedtype ModuleType: Module
    
    /// Flag of if multiple GPUs will be used for the model
    var useMultiGPUs: Bool { get }
    
    /// The target device of model
    var device: Device { get }
    
    /// The real model to run
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
        var no_grad = NoGrad()
        model.eval()
        
        // validate
        let valResultList: [String: Array<Float>] = try with(&no_grad) { _ in
            // initialize validation
            var resultList: [String: Array<Float>] = [:]
            
            // batch loop
            for (batch, example) in dataLoader.enumerated() {
                // extract example
                var xTest = Tensor(example.tuple2.0)
                var yTest = Tensor(example.tuple2.1)
                if useMultiGPUs != true { xTest.to(device) }
                yTest.to(device)
                
                // train step
                let result = valStep(xTest, yTest)
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
        } as! [String : Array<Float>]
    
        // initialize val result
        var valResult: [String: Float] = [:]
        
        // calculate mean
        for (key, m) in valResultList {
            valResult[key] = Float(torch.Tensor(m).mean())!
        }
        
        // reset model mode
        return valResult
    }
    
    /// Validation for a single step
    /// - Parameters:
    ///   - input: A torch.Tensor for input
    ///   - label: A torch.Tensor for label
    /// - Returns: A Dictionary of result of validation
    func valStep(_ xTest: Tensor, _ yTest: Tensor) -> [String: Float] {
        // forward pass
        let y = model(xTest)
        let loss = calculateLoss(yTrue: yTest, yPred: y)
        var metrics = calculateMetrics(yTrue: yTest, yPred: y)
        metrics["loss"] = Float(loss)!
        
        // backward pass
        return metrics
    }
}

/// Main Training Protocol
public protocol TrainingManager: EvaluatingManager {
    /// The learning rate scheduler type
    associatedtype LrSchedulerType: LrScheduler
    
    /// The optimizer type
    associatedtype OptimizerType: Optimizer
    
    /// The learning rate schedule which updates the learning rate in optimizer
    var lrScheduler: LrSchedulerType? { get set }
    
    /// The optimizer to update model
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
            model.train()
            
            // batch loop
            for (batch, example) in trainingDatasetLoader.enumerated() {
                // extract example
                var xTrain = Tensor(example.tuple2.0)
                var yTrain = Tensor(example.tuple2.1)
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
        metrics["loss"] = Float(loss)!
        
        // backward pass
        backward(loss)
        return metrics
    }
}