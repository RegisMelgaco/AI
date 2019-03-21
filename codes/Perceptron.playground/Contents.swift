import Cocoa

var str = "Hello, playground"

class SimplePerceptron {
    
    var weights: [Double]
    var bias: Double
    let inputSize: Int
    var learningRate: Double
    var age: Int
    var maxAge: Int
    
    init(inputSize: Int, learningRate: Double, initialMinWeight: Double, initialMaxWeight: Double, maxAge: Int = 10000) {
        self.inputSize = inputSize
        self.learningRate = learningRate
        self.age = 0
        self.maxAge = maxAge
        self.bias = Double.random(in: initialMinWeight...initialMaxWeight)
        self.weights = []
        
        for _ in 1...inputSize {
            self.weights.append(Double.random(in: initialMinWeight...initialMaxWeight))
        }
    }
    
    func activation (input: [Double]) -> Double {
        var acc: Double = self.bias
        for i in 0..<input.count {
            acc += input[i] * self.weights[i]
        }
        return acc
    }
    
    func updateWeightsAndBias(input: [Double], error: Double) {
        bias += learningRate * error
        for i in 0..<self.weights.count {
            self.weights[i] += input[i] * error * learningRate
        }
    }
    
    func predict (input: [Double]) -> Double {
        if (self.activation(input: input) >= 0) {
            return 1.0
        } else {
            return 0.0
        }
    }
    
    func fit (inputs: [[Double]], labels: [Double]) {
        var cases: [Case] = []
        for i in 0..<labels.count {
            cases.append(Case(input: inputs[i], label: labels[i]))
        }
        
        var error: Double
        var i = 0
        var loopsSinceLastError = 0
        age = 0
        while (loopsSinceLastError <= cases.count && age < maxAge) {
            if (i == 0) {
                cases = cases.shuffled()
                loopsSinceLastError = 0
            }
            
            error = cases[i].label - predict(input: cases[i].input)
            
            if (error != 0.0) {
                loopsSinceLastError = 0
                updateWeightsAndBias(input: cases[i].input, error: error)
            } else {
                loopsSinceLastError += 1
            }
            
            age += 1
            i = age % cases.count
        }
    }
}

class Case {
    let input: [Double]
    let label: Double
    
    init (input: [Double], label: Double){
        self.input = input
        self.label = label
    }
}

let sp = SimplePerceptron(inputSize: 2, learningRate: 0.1, initialMinWeight: -2.0, initialMaxWeight: -2.0)
//sp.weights = [1,1]
//sp.bias = -1.5

let inputs: [[Double]] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
let labels: [Double] = [0.0, 0.0, 0.0, 1.0]

sp.fit(inputs: inputs, labels: labels)

sp.weights
sp.bias
sp.age

sp.predict(input: [0.0, 0.0])
sp.predict(input: [1.0, 0.0])
sp.predict(input: [1.0, 1.0])
