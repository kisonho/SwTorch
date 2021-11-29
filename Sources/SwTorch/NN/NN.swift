//
//  NN.swift
//  
//
//  Created by Kison Ho on 11/11/21.
//

import Foundation
import PythonKit

fileprivate let Parameter = Python.import("torch.nn.parameter.Parameter")
fileprivate let F = Python.import("torch.nn.functional")

/// Main conv2d module
struct Conv2D: WeightedModule {
    var bias: Tensor? = nil
    
    /// The `Int` of dilations
    let dilation: Int
    
    /// The `Int` of input channel groups
    let groups: Int
    
    /// Number of input features
    var inFeatures: Int { get {
        return weight.shape[1] * groups
    }}
    
    /// Padding method
    let padding: Padding
    
    /// The stride of the conv kernel
    let stride: (h: Int, w: Int)
    
    var weight: Tensor
    
    init(_ file: URL) {
        // load convolutional layers
        let conv2d: [String: PythonObject?] = Dictionary(torch.load(file.absoluteString))!
        
        // load parameters
        self.bias = conv2d["bias"] != nil ? Tensor(conv2d["bias"]!!) : nil
        self.dilation = Int(conv2d["dilation"]!!)!
        self.groups = Int(conv2d["groups"]!!)!
        self.weight = Tensor(conv2d["weight"]!!)
        
        // load padding
        switch String(conv2d["padding"]!!)! {
        case "valid":
            self.padding = .valid
        case "same":
            self.padding = .same
        default:
            print("[Warning]: Padding with \'\(conv2d["padding"]!!)\' is currently not supported, using \'same\' instead.")
            self.padding = .same
        }
        
        // load stride
        let stride = conv2d["stride"]!!.tuple2
        self.stride = (h: Int(stride.0)!, w: Int(stride.1)!)
    }
    
    /// Main Constructor
    /// - Parameters:
    ///   - inChannels: An `Int` of input channels
    ///   - outChannels: An `Int` of output channels
    ///   - kernelSize: A `Tuple<Int, Int>` of kernel size
    ///   - stride: A `Tuple<Int, Int>` of strides
    ///   - padding: A `Padding` method
    ///   - dilation: An `Int` of dilations
    ///   - groups: An `Int` of input channel groups
    ///   - bias: A `Bool` flag of if using bias
    ///   - activation: Activation function that accepts an input `Tensor` and give an output `Tensor`
    /// - Throws: `ModuleError.invalidGroups` when input channels cannot be exactly groupped into groups given
    init(inChannels: Int, outChannels: Int, kernelSize: (h: Int, w: Int), stride: (h: Int, w: Int) = (h: 1, w: 1), padding: Padding = .same, dilation: Int = 1, groups: Int = 1, bias: Bool = true) throws {
        // initialize parameters
        guard inChannels % groups == 0 else { throw ModuleError.invalidGroups }
        self.bias = bias == true ? Tensor(Parameter(torch.empty(outChannels))) : nil
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.weight = Tensor(Parameter(torch.empty([outChannels, inChannels / groups, kernelSize.h, kernelSize.w])))
    }
    
    func copy() -> Conv2D {
        // initialize copy
        var newConv = try! Conv2D(inChannels: inFeatures, outChannels: outFeatures, kernelSize: (h: weight.shape[2], w: weight.shape[3]), stride: stride, padding: padding, dilation: dilation, groups: groups, bias: bias != nil)
        newConv.bias = bias != nil ? Tensor(value: bias) : nil
        newConv.weight = Tensor(value: weight)
        return newConv
    }
    
    func eval() {
        return
    }
    
    func forward(_ x: Tensor) -> Tensor {
        Tensor(F.conv2d(x, weight, bias, stride: [stride.h, stride.w], padding: padding.rawValue, dilation: dilation, groups: groups))
    }
    
    func train() {
        return
    }
    
    func save(_ file: URL) {
        let stateDict: [String: PythonObject?] = ["bias": bias != nil ? PythonObject(bias) : nil,
                                                  "dilation": PythonObject(dilation),
                                                  "groups": PythonObject(groups),
                                                  "padding": PythonObject(padding.rawValue),
                                                  "stride": PythonObject([stride.h, stride.w]),
                                                  "weight": PythonObject(weight)]
        torch.save(stateDict, file.absoluteString)
    }
}

/// A module to flatten tensor
public struct Flatten: Module {
    /// Last dim to flatten
    let endDim: Int
    
    /// First dim to flatten
    let startDim: Int
    
    public init(_ file: URL) {
        let flatten: [String: PythonObject?] = Dictionary(torch.load(file.absoluteString))!
        self.startDim = Int(flatten["start_dim"]!!)!
        self.endDim = Int(flatten["end_dim"]!!)!
    }
    
    /// Constructor
    /// - Parameters:
    ///   - startDim: An `Int` of first dim to flatten
    ///   - endDim: An `Int` of last dim to flatten
    public init(startDim: Int = 1, endDim: Int = -1) {
        self.startDim = startDim
        self.endDim = endDim
    }
    
    public func copy() -> Flatten {
        Flatten(startDim: startDim, endDim: endDim)
    }
    
    public func eval() {
        return
    }
    
    public func forward(_ x: Tensor) -> Tensor {
        x.flatten(startDim: startDim, endDim: endDim)
    }
    
    mutating public func loadStateDict(_ dict: [String : PythonObject?]) {
        return
    }
    
    public func train() {
        return
    }
    
    public func save(_ file: URL) {
        let stateDict: [String: PythonObject?] = ["start_dim": PythonObject(startDim),
                                                  "end_dim": PythonObject(endDim)]
        torch.save(stateDict, file.absoluteString)
    }
}

/// Main linear module
struct Linear: WeightedModule {
    var bias: Tensor? = nil
    
    var weight: Tensor
    
    init(_ file: URL) {
        // load convolutional layers
        let linear: [String: PythonObject?] = Dictionary(torch.load(file.absoluteString))!
        self.bias = linear["bias"] != nil ? Tensor(linear["bias"]!!) : nil
        self.weight = Tensor(linear["weight"]!!)
    }
    
    /// Constructor
    /// - Parameters:
    ///   - inFeatures: An `Int` of input features
    ///   - outFeatures: An `Int` of output features
    ///   - bias: A `Bool` flag of if using bias
    init(inFeatures: Int, outFeatures: Int, bias: Bool = true) {
        // initialize bias
        if bias == true {
            self.bias = Tensor(Parameter(torch.empty(outFeatures)))
        }
        
        // initialize weight
        self.weight = Tensor(Parameter(torch.empty([outFeatures, inFeatures])))
    }
    
    func copy() -> Linear {
        var newLinear = Linear(inFeatures: inFeatures, outFeatures: outFeatures, bias: bias != nil)
        newLinear.bias = bias != nil ? Tensor(value: bias) : nil
        newLinear.weight = Tensor(value: weight)
        return newLinear
    }
    
    func eval() {
        return
    }
    
    func forward(_ x: Tensor) -> Tensor {
        return Tensor(F.linear(x, weight, bias))
    }
    
    func train() {
        return
    }
    
    func save(_ file: URL) {
        let stateDict: [String: PythonObject?] = ["bias": bias != nil ? PythonObject(bias) : nil,
                                                  "weight": PythonObject(weight)]
        torch.save(stateDict, file.absoluteString)
    }
}

/// Possible errors occured in modules
enum ModuleError: Error {
    case invalidGroups
}

/// Supported padding methods
enum Padding: String {
    case valid = "valid"
    case same = "same"
}

/// A module with bias and weight
protocol WeightedModule: DeviceMovable, Module {
    /// The bias `Tensor`
    var bias: Tensor? { get set }
    
    /// The weight `Tensor`
    var weight: Tensor { get set }
}

extension WeightedModule {
    /// Number of input features
    var inFeatures: Int { get {
        return weight.shape[1]
    }}
    
    /// Number of output features
    var outFeatures: Int { get {
        return weight.shape.first!
    }}
    
    /// get all the parameters in target module
    var parameters: Array<Tensor> { get {
        if bias != nil { return [weight, bias!]}
        else { return [weight] }
    }}
    
    mutating func loadStateDict(_ dict: [String : PythonObject?]) {
        let bias = dict["bias"]!
        self.bias = bias != nil ? Tensor(bias!) : nil
        self.weight = Tensor(dict["weight"]!!)
    }
    
    mutating func to(_ device: Device, id: Int? = nil) {
        bias?.to(device, id: id)
        weight.to(device, id: id)
    }
}
