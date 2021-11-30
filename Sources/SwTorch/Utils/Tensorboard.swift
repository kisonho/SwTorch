//
//  File.swift
//  
//
//  Created by Kison Ho on 11/30/21.
//

import Foundation
import PythonKit

fileprivate let tensorboard = Python.import("torch.utils.tensorboard")

/// A summary writer that write data into Tensorboard
public class SummaryWriter: Enterable {
    /// The log directory of writer
    var logDir: URL?
    
    /// The python summary writer pointer
    var writerPtr: PythonObject? = nil
    
    /// Constructor
    /// - Parameter logDir: An optional `URL` of the target writer
    public init(logDir: URL? = nil) {
        self.logDir = logDir
    }
    
    /// Add scalar to the board
    /// - Parameters:
    ///   - name: A `String` of scalar name
    ///   - data: A `Float` of data to be recorded
    ///   - iter: An `Int` of iteration index
    public func addScalar(_ name: String, _ data: Float, iter: Int) {
        writerPtr!.add_scalar(name, data, iter)
    }
    
    /// Add scalar to the board
    /// - Parameters:
    ///   - mainTag: A `String` of scalar name
    ///   - data: A `Float` of data to be recorded
    ///   - iter: An `Int` of iteration index
    public func addScalars(_ mainTag: String, _ data: [String: Float], iter: Int) {
        writerPtr!.add_scalar(mainTag, data, iter)
    }
    
    public func enter() {
        writerPtr = tensorboard.SummaryWriter(logDir?.path)
    }
    
    public func exit() {
        writerPtr?.close()
    }
}
