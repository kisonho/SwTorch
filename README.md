# SwTorch
PyTorch training/evaluation wrap for Swift.
---
## Requirements
*The latest LTS version of [PyTorch](https://pytorch.org) with python environment set up to `PYTHON_LIBRARY`
* Add packages:
```
.package(url: "https://github.com/kisonho/SwTorch.git", branch: "main"),
```
* Swift Dependencies
    * [PythonKit](https://github.com/pvieito/PythonKit.git)
---
## Creating a Module
Create a model that implements the **Module** protocol, or load a python module directly with struct **PyModule**.

## PyTorch Tensor representation
A PyTorch Tensor (`torch.nn.Tensor`) is represented as Tensor, which actual variable was stored in its `pythonObject`. 

## Evaluation
* Creating a class to implement the **Evaluating** protocol
```
import SwTorch

class ModelEvaluator: Evaluating {
    ...
}
```
* Call `validate(<DataLoader>)` to evaluate the model. The method will returns a Dictionary of metrics that averaged by returned metrics of `calculatedMetrics(yTrue, yPred)`.
* `onBatchEnd(batch, result)` is the call back method for each batch validation.

## Training
* Creating a class to implement the **Training** protocol
```
import SwTorch

class ModelEvaluator: Training {
    ...
}
```
* Call `train(trainingDataset: <DataLoader>, epochs: <Int>, initialEpoch: <Int=0>， validationDataset: <DataLoader?=nil>)` to train the model. The method will return a Dictionary of best validation result among the training epochs, or a `nil` if `validationDataset` is not given.

## Device Management
Use `.to(<Device>)` to move a `DeviceMovable` object to target device.

## Unmapped PyTorch operations or types
Directly use `torch.*` as `PythonObject`

## MNIST Example
* A Simple CNN example for MNIST dataset:
[MNISTExample](https://github.com/kisonho/MNIST-SwTorch-Example)
