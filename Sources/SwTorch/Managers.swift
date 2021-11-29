//
//  Managers.swift
//
//
//  Created by Kison Ho on 10/26/21.
//

import PythonKit

/// import required modules
public let torch = Python.import("torch")

/// Main Validation Protocol
public protocol Evaluating {
    /// Flag of if multiple GPUs will be used for the model
    var useMultiGPUs: Bool { get }
    
    /// The target device of model
    var device: Device { get }
    
    /// On validation starts
    func onValStart()
    
    /// Validation for a single step
    /// - Parameters:
    ///   - input: A torch.Tensor for input
    ///   - label: A torch.Tensor for label
    /// - Returns: A Dictionary of result of validation
    func valStep(_ xTest: Tensor, _ yTest: Tensor) -> [String: Float]
}

public extension Evaluating {
    /// Main validation function
    /// - Parameter dataLoader: A torch.utils.data.DataLoader to load the dataset
    /// - Returns: A Dictionary of validation results
    func validate(_ dataset: PythonObject) throws -> [String: Float] {
        // no gradients
        onValStart()
        var no_grad = NoGrad()
        
        // validate
        let valResultList: [String: Array<Float>] = try with(&no_grad) { _ in
            // initialize validation
            var resultList: [String: Array<Float>] = [:]
            
            // batch loop
            for (batch, example) in dataset.enumerated() {
                // extract example
                var xTest = Tensor(example.tuple2.0)
                var yTest = Tensor(example.tuple2.1)
                if useMultiGPUs != true { xTest.to(device) }
                yTest.to(device)
                
                // train step
                let result = valStep(xTest, yTest)
                
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
            valResult[key] = Float(Tensor(value: m).mean())!
        }
        
        // reset model mode
        return valResult
    }
}

/// Main Training Protocol
public protocol Training: Evaluating {
    /// On every batch ends
    func onBatchEnd(batch: Int, result: [String: Float])
    
    /// On every epoch starts
    func onEpochStart(epoch: Int, totalEpochs: Int)
    
    /// On every epoch ends
    /// - Returns: A Bool of flag if this result is best
    func onEpochEnd(epoch: Int, totalEpochs: Int, trainingResult: [String: Float], valResult: [String: Float]?) -> Bool
    
    /// Train for one step
    /// - Parameters:
    ///   - input: A torch.Tensor for input
    ///   - label: A torch.Tensor for label
    /// - Returns: A Dictionary of results
    func trainStep(_ xTrain: Tensor, _ yTrain: Tensor) -> [String: Float]
}

public extension Training {
    /// Main training function
    /// - Parameters:
    ///   - trainingDataset: A torch.utils.data.DataLoader to load the training dataset
    ///   - epochs: An Int of number of epochs
    ///   - initialEpoch: An Int of the starting epoch index
    ///   - validationDataset: An optional torch.utils.data.DataLoader to load validation dataset
    /// - Returns: An optional Dictionary of metrics (if validationDatasetLoader is not nil)
    mutating func train(trainingDataset: PythonObject, epochs: Int, initialEpoch: Int = 0, validationDataset: PythonObject? = nil) throws -> [String: Float]? {
        // initialize training
        var bestResult: [String: Float]? = nil
        
        // check initial epoch to be smaller then epochs
        if initialEpoch >= epochs {
            print("[Warning]: initial epochs \(initialEpoch) is larger or equal then epochs \(epochs)!")
            return nil
        }
        
        // epoch loop
        for epoch in initialEpoch ..< epochs {
            // initialize epoch
            var resultList: [String: Array<Float>] = [:]
            onEpochStart(epoch: epoch, totalEpochs: epochs)
            
            // batch loop
            for (batch, example) in trainingDataset.enumerated() {
                // extract example
                var xTrain = Tensor(example.tuple2.0)
                var yTrain = Tensor(example.tuple2.1)
                
                // move to device
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
            
            // calculate mean
            for (key, m) in resultList {
                trainingResult[key] = Float(Tensor(value: m).mean())!
            }
            
            // validation
            let valResult = validationDataset != nil ? try self.validate(validationDataset!) : nil
            
            // end epoch callback
            let isBest = self.onEpochEnd(epoch: epoch, totalEpochs: epochs, trainingResult: trainingResult, valResult: valResult)
            
            // set best result
            bestResult = isBest == true ? valResult : bestResult
        }
        
        return bestResult
    }
}
