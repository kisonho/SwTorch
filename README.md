# # SwTorch
PyTorch training/evaluation wrap for Swift.
---
## Requirements
*[PyTorch](https://pytorch.org) with python environment set up to `PYTHON_LIBRARY`
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
* Creating a class to implement the **EvaluatingManager**
```
import SwTorch

class ModelEvaluator: EvaluatingManager {
    ...
}
```
* Call `validate(<DataLoader>)` to evaluate the model. The method will returns a Dictionary of metrics that averaged by returned metrics of `calculatedMetrics(yTrue, yPred)`.
* `onBatchEnd(batch, result)` is the call back method for each batch validation.

## Training
* Creating a class to implement the **TrainingManager**
```
import SwTorch

class ModelEvaluator: EvaluatingManager {
    ...
}
```
* Call `train(trainingDatasetLoader: <DataLoader>, epochs: <Int>, initialEpoch: <Int=0>， validationDatasetLoader: <DataLoader?=nil>)` to train the model. The method will return a Dictionary of best validation result among the training epochs, or a `nil` if `validationDatasetLoader` is not given.

## Device Management
Use `.to(<Device>)` to move a `DeviceMovable` object to target device.
