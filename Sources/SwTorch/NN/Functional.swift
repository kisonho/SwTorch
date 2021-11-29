//
//  File.swift
//  
//
//  Created by Kison Ho on 11/29/21.
//

import PythonKit

/// core python library
fileprivate let F = Python.import("torch.nn.functional")

public func logSoftmax(_ input: Tensor, dim: Int = 0) -> Tensor {
    return Tensor(F.log_softmax(input, dim: dim))
}

/// Softmax function
/// - Parameters:
///   - input: input `Tensor`
///   - dim: The `Int` of operation dimension
/// - Returns: The softmax result `Tensor`
public func softmax(_ input: Tensor, dim: Int = 0) -> Tensor {
    return Tensor(F.softmax(input, dim: dim))
}
