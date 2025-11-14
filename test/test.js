const assert = require('assert')
const Convolution1DLayer = require('../src/cnn/Convolution1DLayer')
const MaxPool1DLayer = require('../src/cnn/MaxPool1DLayer')
const FullyConnectedLayer = require('../src/cnn/FullyConnectedLayer')
const ReluLayer = require('../src/cnn/ReluLayer')
const SigmoidLayer = require('../src/cnn/SigmoidLayer')
const TanhLayer = require('../src/cnn/TanhLayer')
const SoftmaxLayer = require('../src/cnn/SoftmaxLayer')
const RegressionLayer = require('../src/cnn/RegressionLayer')
const NeuralNetwork = require('../src/cnn/NeuralNetwork')
const Volume = require('../src/cnn/Volume')
const {sigmoid} = require('../src/cnn/utils')

describe('Convolution 1D', function () {
    it('should compute correct shapes', function () {
        const layer = new Convolution1DLayer({nbKernels: 5, kernelSize: 3, stride: 1, inputShape: [4, 1, 2]})

        assert.equal(layer.weights.length, 5)
        assert.equal(layer.weights[0].width, 3)
        assert.equal(layer.weights[0].depth, 2)
        assert.equal(layer.biases.width, 5)
        assert.deepEqual(layer.getOutputShape(), [2, 1, 5])
    })

    it('should throw an error on wrong kernel size / stride compared to input shape', function () {
        assert.throws(function () {
            new Convolution1DLayer({nbKernels: 2, kernelSize: 3, stride: 2, inputShape: [2, 1, 4]})
        }, Error, 'Output shape must be integers, [2,1.5] calculated.')
    })

    it('should throw an error on 2 dimensional inputs', function () {
        assert.throws(function () {
            new Convolution1DLayer({nbKernels: 2, kernelSize: 3, stride: 2, inputShape: [2, 2, 4]})
        }, Error, 'Conv1D Layer can only take input in one dimension (width). Input height should be 1, 4 given.')
    })

    it('should compute padding to conserve shape', function() {
        const layer = new Convolution1DLayer({nbKernels: 5, kernelSize: 3, stride: 1, conserveShape: true, inputShape: [4, 1, 2]})

        assert.equal(layer.padding, 1)
        assert.equal(layer.getOutputShape()[0], 4)
    })

    it('should feed forward', function () {
        const layer = new Convolution1DLayer({
            nbKernels: 2, kernelSize: 3, stride: 1, padding: 1, inputShape: [4, 1, 2],
            weights: [new Volume([[1, 3, 2], [1, -3, 0]]), new Volume([[0, 1, -2], [4, -1, -3]])],
            biases: new Volume([-0.5, 4])
        })

        layer.feedForward(new Volume([[2, 3, 4, 5], [1, 2, 3, 4]]))

        assert.deepEqual(layer.output.data[0][0], Float64Array.from([3*2 + 2*3 + -3 + -0.5, 13.5, 17.5, 4 + 3*5 + 3 + -3*4 + -0.5]))
        assert.deepEqual(layer.output.data[1][0], Float64Array.from([2 + -2*3 + -1 + -3*2 + 4, -8, -9, 5 + 4*3 + -1*4 + 4]))
    })

    it('should back propagate', function () {
        const input = [[2, 3, 4, 5], [1, 2, 3, 4]]
        const inputVolume = new Volume(input)
        const weights = [
            [[1, 3, 2], [1, -3, 0.5]],
            [[0.3, 1, -2], [4, -1, -3]]
        ]
        const layer = new Convolution1DLayer({
            nbKernels: 2, kernelSize: 3, stride: 1, padding: 1, inputShape: [4, 1, 2],
            weights: [new Volume(weights[0]), new Volume(weights[1])],
            biases: new Volume([-0.5, 4])
        })
        const outputGradients = [[0.25, -0.2, 0.7, 0.32], [0.52, -1, 2.3, 0.23]]

        layer.feedForward(inputVolume)
        layer.output.gradients[0][0] = outputGradients[0]
        layer.output.gradients[1][0] = outputGradients[1]
        layer.backPropagate()

        const expectedInputGradients = [
            [
                outputGradients[0][0] * weights[0][0][1] + outputGradients[1][0] * weights[1][0][1] + outputGradients[0][1] * weights[0][0][0] + outputGradients[1][1] * weights[1][0][0],
                outputGradients[0][0] * weights[0][0][2] + outputGradients[1][0] * weights[1][0][2] + outputGradients[0][1] * weights[0][0][1] + outputGradients[1][1] * weights[1][0][1] + outputGradients[0][2] * weights[0][0][0] + outputGradients[1][2] * weights[1][0][0],
                outputGradients[0][1] * weights[0][0][2] + outputGradients[1][1] * weights[1][0][2] + outputGradients[0][2] * weights[0][0][1] + outputGradients[1][2] * weights[1][0][1] + outputGradients[0][3] * weights[0][0][0] + outputGradients[1][3] * weights[1][0][0],
                outputGradients[0][2] * weights[0][0][2] + outputGradients[1][2] * weights[1][0][2] + outputGradients[0][3] * weights[0][0][1] + outputGradients[1][3] * weights[1][0][1],
            ],
            [
                outputGradients[0][0] * weights[0][1][1] + outputGradients[1][0] * weights[1][1][1] + outputGradients[0][1] * weights[0][1][0] + outputGradients[1][1] * weights[1][1][0],
                outputGradients[0][0] * weights[0][1][2] + outputGradients[1][0] * weights[1][1][2] + outputGradients[0][1] * weights[0][1][1] + outputGradients[1][1] * weights[1][1][1] + outputGradients[0][2] * weights[0][1][0] + outputGradients[1][2] * weights[1][1][0],
                outputGradients[0][1] * weights[0][1][2] + outputGradients[1][1] * weights[1][1][2] + outputGradients[0][2] * weights[0][1][1] + outputGradients[1][2] * weights[1][1][1] + outputGradients[0][3] * weights[0][1][0] + outputGradients[1][3] * weights[1][1][0],
                outputGradients[0][2] * weights[0][1][2] + outputGradients[1][2] * weights[1][1][2] + outputGradients[0][3] * weights[0][1][1] + outputGradients[1][3] * weights[1][1][1],
            ],
        ]
        assert.deepEqual(inputVolume.gradients[0][0].map(val => val.toFixed(2)), expectedInputGradients[0].map(val => val.toFixed(2)))
        assert.deepEqual(inputVolume.gradients[1][0].map(val => val.toFixed(2)), expectedInputGradients[1].map(val => val.toFixed(2)))
        const expectedBiasGradients = [0.25 + -0.2 + 0.7 + 0.32, 0.52 + -1 + 2.3 + 0.23]
        assert.deepEqual(layer.biases.gradients[0][0], expectedBiasGradients)
        const expectedWeightGradients = [
            [outputGradients[0][1] * input[0][0] + outputGradients[0][2] * input[0][1] + outputGradients[0][3] * input[0][2], outputGradients[0][0] * input[0][0] + outputGradients[0][1] * input[0][1] + outputGradients[0][2] * input[0][2] + outputGradients[0][3] * input[0][3], outputGradients[0][0] * input[0][1] + outputGradients[0][1] * input[0][2] + outputGradients[0][2] * input[0][3]],
            [outputGradients[0][1] * input[1][0] + outputGradients[0][2] * input[1][1] + outputGradients[0][3] * input[1][2], outputGradients[0][0] * input[1][0] + outputGradients[0][1] * input[1][1] + outputGradients[0][2] * input[1][2] + outputGradients[0][3] * input[1][3], outputGradients[0][0] * input[1][1] + outputGradients[0][1] * input[1][2] + outputGradients[0][2] * input[1][3]],
            [outputGradients[1][1] * input[0][0] + outputGradients[1][2] * input[0][1] + outputGradients[1][3] * input[0][2], outputGradients[1][0] * input[0][0] + outputGradients[1][1] * input[0][1] + outputGradients[1][2] * input[0][2] + outputGradients[1][3] * input[0][3], outputGradients[1][0] * input[0][1] + outputGradients[1][1] * input[0][2] + outputGradients[1][2] * input[0][3]],
            [outputGradients[1][1] * input[1][0] + outputGradients[1][2] * input[1][1] + outputGradients[1][3] * input[1][2], outputGradients[1][0] * input[1][0] + outputGradients[1][1] * input[1][1] + outputGradients[1][2] * input[1][2] + outputGradients[1][3] * input[1][3], outputGradients[1][0] * input[1][1] + outputGradients[1][1] * input[1][2] + outputGradients[1][2] * input[1][3]],
        ]
        assert.deepEqual(layer.weights[0].gradients[0][0], expectedWeightGradients[0])
        assert.deepEqual(layer.weights[0].gradients[1][0], expectedWeightGradients[1])
        assert.deepEqual(layer.weights[1].gradients[0][0], expectedWeightGradients[2])
        assert.deepEqual(layer.weights[1].gradients[1][0], expectedWeightGradients[3])

        const input2 = new Volume([[2, 3, 4, 5], [1, 2, 3, 4]])

        layer.feedForward(input2)
        layer.output.gradients[0][0] = outputGradients[0]
        layer.output.gradients[1][0] = outputGradients[1]
        layer.backPropagate()

        assert.deepEqual(input2.gradients[0][0].map(val => val.toFixed(2)), expectedInputGradients[0].map(val => val.toFixed(2)))
        assert.deepEqual(input2.gradients[1][0], expectedInputGradients[1])
        assert.deepEqual(layer.biases.gradients[0][0], expectedBiasGradients.map(value => 2 * value))
        assert.deepEqual(layer.weights[0].gradients[0][0].map(val => val.toFixed(2)), expectedWeightGradients[0].map(value => 2 * value).map(val => val.toFixed(2)))
        assert.deepEqual(layer.weights[0].gradients[1][0].map(val => val.toFixed(2)), expectedWeightGradients[1].map(value => 2 * value).map(val => val.toFixed(2)))
        assert.deepEqual(layer.weights[1].gradients[0][0].map(val => val.toFixed(2)), expectedWeightGradients[2].map(value => 2 * value).map(val => val.toFixed(2)))
        assert.deepEqual(layer.weights[1].gradients[1][0].map(val => val.toFixed(2)), expectedWeightGradients[3].map(value => 2 * value).map(val => val.toFixed(2)))
    })
})

describe('MaxPool 1D', function () {
    it('should compute correct shapes', function () {
        const layer = new MaxPool1DLayer({kernelSize: 2, stride: 2, inputShape: [84, 1, 2]})

        assert.deepEqual(layer.getOutputShape(), [42, 1, 2])
    })

    it('should throw an error on wrong kernel size / stride compared to input shape', function () {
        assert.throws(function () {
            new MaxPool1DLayer({kernelSize: 3, stride: 2, inputShape: [4, 1, 2]})
        }, Error, 'Output shape must be integers, [1.5,1,2] calculated.')
    })

    it('should feed forward', function () {
        const input = new Volume([[1.5, 3, 2, 2.7, 4, 1.8, 2, 3, 2, 5], [3, 7, 1, 5, 2, 8, 4, 1, 0, 8]])
        const layer = new MaxPool1DLayer({kernelSize: 2, stride: 2, inputShape: input.getShape()})

        layer.feedForward(input)

        assert.deepEqual(layer.output.data[0][0], [3, 2.7, 4, 3, 5])
        assert.deepEqual(layer.output.data[1][0], [7, 5, 8, 4, 8])
    })

    it('should back propagate', function () {
        const input = new Volume([[1.5, 3, 2, 2.7, 4, 1.8, 2, 3, 2, 5], [3, 7, 1, 5, 2, 8, 4, 1, 0, 8]])
        const layer = new MaxPool1DLayer({kernelSize: 2, stride: 2, inputShape: input.getShape()})
        const outputGradients = [[-0.2, 0.7, 1.4, -5.2, 0.12], [-1, 2.3, 1.5, -0.45, 0.34]]

        layer.feedForward(input)
        layer.output.gradients[0][0] = outputGradients[0]
        layer.output.gradients[1][0] = outputGradients[1]
        layer.backPropagate()

        assert.deepEqual(input.gradients[0][0], [0, -0.2, 0, 0.7, 1.4, 0, 0, -5.2, 0, 0.12])
        assert.deepEqual(input.gradients[1][0], [0, -1, 0, 2.3, 0, 1.5, -0.45, 0, 0, 0.34])
    })

    it('should feed forward with overlap', function () {
        const input = new Volume([[1.5, 3, 2, 2.7, 4, 1.8, 2], [3, 7, 1, 5, 2, 8, 4]])
        const layer = new MaxPool1DLayer({kernelSize: 3, stride: 2, inputShape: input.getShape()})

        layer.feedForward(input)

        assert.deepEqual(layer.output.data[0][0], [3, 4, 4])
        assert.deepEqual(layer.output.data[1][0], [7, 5, 8])
    })

    it('should back propagate with overlap', function () {
        const input = new Volume([[1.5, 3, 2, 2.7, 4, 1.8, 2], [3, 7, 1, 5, 2, 8, 4]])
        const layer = new MaxPool1DLayer({kernelSize: 3, stride: 2, inputShape: input.getShape()})
        const outputGradients = [[-0.2, 0.7, 1.4], [-1, 2.3, 1.5]]

        layer.feedForward(input)
        layer.output.gradients[0][0] = outputGradients[0]
        layer.output.gradients[1][0] = outputGradients[1]
        layer.backPropagate()

        assert.deepEqual(input.gradients[0][0].map(val => val.toFixed(1)), [0, -0.2, 0, 0, 2.1, 0, 0])
        assert.deepEqual(input.gradients[1][0], [0, -1, 0, 2.3, 0, 1.5, 0])
    })
})

describe('Fully connected', function () {
    it('should compute correct shapes', function () {
        const layer = new FullyConnectedLayer({nbNeurons: 5, inputShape: [2, 3, 4]})

        assert.equal(layer.weights.length, 5)
        assert.equal(layer.weights[0].width, 2 * 3 * 4)
        assert.equal(layer.biases.width, 5)
        assert.deepEqual(layer.getOutputShape(), [5, 1, 1])
    })

    it('should feed forward', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const layer = new FullyConnectedLayer({
            nbNeurons: 3,
            inputShape: input.getShape(),
            weights: [
                new Volume([0.5, 1, 1.5, -0.5, -1, 0.75, -0.2, 0.25]),
                new Volume([0.1, -0.3, 0.4, 1, -0.5, 0.7, -0.25, 1.25]),
                new Volume([-0.35, 0.3, -1, 0.25, 1, 0.2, 0.3, -0.75])
            ],
            biases: new Volume([1, 2, -1])
        })

        const output = layer.feedForward(input)

        assert.deepEqual(output.data[0][0], [-6.1, 6.0375, 2.575])
    })

    it('should back propagate', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const layer = new FullyConnectedLayer({
            nbNeurons: 3,
            inputShape: input.getShape(),
            weights: [
                new Volume([0.5, 1, 1.5, -0.5, -1, 0.75, -0.2, 0.25]),
                new Volume([0.1, -0.3, 0.4, 1, -0.5, 0.7, -0.25, 1.25]),
                new Volume([-0.35, 0.3, -1, 0.25, 1, 0.2, 0.3, -0.75])
            ],
            biases: new Volume([1, 2, -1])
        })
        const outputGradients = [-0.2, 0.7, 1.4]

        layer.feedForward(input)
        layer.output.gradients[0][0] = outputGradients
        layer.backPropagate()

        const expectedInputGradients = [
            [-0.2 * 0.5 + 0.7 * 0.1 + 1.4 * -0.35, -0.2 * 1 + 0.7 * -0.3 + 1.4 * 0.3, -0.2 * 1.5 + 0.7 * 0.4 + 1.4 * -1, -0.2 * -0.5 + 0.7 * 1 + 1.4 * 0.25],
            [-0.2 * -1 + 0.7 * -0.5 + 1.4 * 1, -0.2 * 0.75 + 0.7 * 0.7 + 1.4 * 0.2, -0.2 * -0.2 + 0.7 * -0.25 + 1.4 * 0.3, -0.2 * 0.25 + 0.7 * 1.25 + 1.4 * -0.75]
        ]
        assert.deepEqual(input.gradients[0][0], expectedInputGradients[0])
        assert.deepEqual(input.gradients[1][0], expectedInputGradients[1])
        assert.deepEqual(layer.biases.gradients[0][0], outputGradients)
        const expectedWeightGradients = [
            [-0.2 * -2, -0.2 * 1, -0.2 * -3, -0.2 * 4, -0.2 * 0.5, -0.2 * -1, -0.2 * -0.75, -0.2 * 2],
            [0.7 * -2, 0.7 * 1, 0.7 * -3, 0.7 * 4, 0.7 * 0.5, 0.7 * -1, 0.7 * -0.75, 0.7 * 2],
            [1.4 * -2, 1.4 * 1, 1.4 * -3, 1.4 * 4, 1.4 * 0.5, 1.4 * -1, 1.4 * -0.75, 1.4 * 2],
        ]
        assert.deepEqual(layer.weights[0].gradients[0][0], expectedWeightGradients[0])
        assert.deepEqual(layer.weights[1].gradients[0][0], expectedWeightGradients[1])
        assert.deepEqual(layer.weights[2].gradients[0][0], expectedWeightGradients[2])

        const input2 = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])

        layer.feedForward(input2)
        layer.output.gradients[0][0] = outputGradients
        layer.backPropagate()

        assert.deepEqual(input2.gradients[0][0], expectedInputGradients[0])
        assert.deepEqual(input2.gradients[1][0], expectedInputGradients[1])
        assert.deepEqual(layer.biases.gradients[0][0], outputGradients.map(value => 2 * value))
        assert.deepEqual(layer.weights[0].gradients[0][0], expectedWeightGradients[0].map(value => 2 * value))
        assert.deepEqual(layer.weights[1].gradients[0][0], expectedWeightGradients[1].map(value => 2 * value))
        assert.deepEqual(layer.weights[2].gradients[0][0], expectedWeightGradients[2].map(value => 2 * value))
    })
})

describe('Softmax', function () {
    it('should feed forward', function () {
        const layer = new SoftmaxLayer({inputShape: [4, 1, 1]})
        const expectedOutput = [0.2368828181, 0.6439142599, 0.03205860328, 0.08714431874]

        layer.feedForward(new Volume([3, 4, 1, 2]))

        for (let i = 0; i < expectedOutput.length; i++) {
            assert.equal(layer.output.data[0][0][i].toFixed(10), expectedOutput[i].toFixed(10))
        }
    })

    it('should back propagate', function () {
        const layer = new SoftmaxLayer({inputShape: [4, 1, 1]})
        const input = new Volume([3, 4, 1, 2])
        const expectedOutput = [0.2368828181, 0.6439142599, 0.03205860328 - 1, 0.08714431874]

        layer.feedForward(input)

        assert.equal(layer.backPropagate(2), -Math.log(layer.output.data[0][0][2]))
        for (let i = 0; i < expectedOutput.length; i++) {
            assert.equal(input.gradients[0][0][i].toFixed(10), expectedOutput[i].toFixed(10))
        }
    })
})

describe('Regression', function () {
    it('should feed forward', function () {
        const input = new Volume([3, 4, 1, 2])
        const layer = new RegressionLayer({inputShape: input.getShape()})
        const expectedOutput = input.data[0][0].slice(0)

        layer.feedForward(input)

        for (let i = 0; i < expectedOutput.length; i++) {
            assert.equal(layer.output.data[0][0][i], expectedOutput[i])
        }
    })

    it('should back propagate', function () {
        const input = new Volume([3, 4, 1, 2])
        const layer = new RegressionLayer({inputShape: input.getShape()})
        const expectedOutput = [0.2, 1, 4.5, 0]

        layer.feedForward(input)
        layer.backPropagate(expectedOutput)

        for (let i = 0; i < expectedOutput.length; i++) {
            assert.equal(input.gradients[0][0][i], input.data[0][0][i] - expectedOutput[i])
        }
    })
})

describe('Relu', function () {
    it('should output correct shape', function () {
        const layer = new ReluLayer({inputShape: [2, 2, 4]})

        assert.deepEqual(layer.getOutputShape(), [2, 2, 4])
    })

    it('should output relued values', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const layer = new ReluLayer({inputShape: input.getShape()})

        layer.feedForward(input)

        assert.deepEqual(layer.output.data[0][0], [0, 1, 0, 4])
        assert.deepEqual(layer.output.data[1][0], [0.5, 0, 0, 2])
    })

    it('should back propagate', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const outputGradients = [[-0.2, 0.7, 1.4, -5.2], [-1, 2.3, 1.5, -0.45]]
        const layer = new ReluLayer({inputShape: input.getShape()})

        layer.feedForward(input)
        layer.output.gradients[0][0] = outputGradients[0]
        layer.output.gradients[1][0] = outputGradients[1]
        layer.backPropagate()

        assert.deepEqual(layer.input.gradients[0][0], [0, 0.7, 0, -5.2])
        assert.deepEqual(layer.input.gradients[1][0], [-1, 0, 0, -0.45])
    })
})

describe('Sigmoid', function () {
    it('should output correct shape', function () {
        const layer = new SigmoidLayer({inputShape: [2, 2, 4]})

        assert.deepEqual(layer.getOutputShape(), [2, 2, 4])
    })

    it('should output sigmoid values', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const layer = new SigmoidLayer({inputShape: input.getShape()})

        layer.feedForward(input)

        assert.deepEqual(layer.output.data[0][0], [sigmoid(-2), sigmoid(1), sigmoid(-3), sigmoid(4)])
        assert.deepEqual(layer.output.data[1][0], [sigmoid(0.5), sigmoid(-1), sigmoid(-0.75), sigmoid(2)])
    })

    it('should back propagate', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const outputGradients = [[-0.2, 0.7, 1.4, -5.2], [-1, 2.3, 1.5, -0.45]]
        const layer = new SigmoidLayer({inputShape: input.getShape()})

        layer.feedForward(input)
        layer.output.gradients[0][0] = outputGradients[0]
        layer.output.gradients[1][0] = outputGradients[1]
        layer.backPropagate()

        assert.deepEqual(input.gradients[0][0], [
            sigmoid(-2) * (1 - sigmoid(-2)) * outputGradients[0][0],
            sigmoid(1) * (1 - sigmoid(1)) * outputGradients[0][1],
            sigmoid(-3) * (1 - sigmoid(-3)) * outputGradients[0][2],
            sigmoid(4) * (1 - sigmoid(4)) * outputGradients[0][3]
        ])
        assert.deepEqual(input.gradients[1][0], [
            sigmoid(0.5) * (1 - sigmoid(0.5)) * outputGradients[1][0],
            sigmoid(-1) * (1 - sigmoid(-1)) * outputGradients[1][1],
            sigmoid(-0.75) * (1 - sigmoid(-0.75)) * outputGradients[1][2],
            sigmoid(2) * (1 - sigmoid(2)) * outputGradients[1][3]
        ])
    })
})

describe('Tanh', function () {
    it('should output correct shape', function () {
        const layer = new TanhLayer({inputShape: [2, 2, 4]})

        assert.deepEqual(layer.getOutputShape(), [2, 2, 4])
    })

    it('should output tanh values', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const layer = new TanhLayer({inputShape: input.getShape()})

        layer.feedForward(input)

        assert.deepEqual(layer.output.data[0][0], [Math.tanh(-2), Math.tanh(1), Math.tanh(-3), Math.tanh(4)])
        assert.deepEqual(layer.output.data[1][0], [Math.tanh(0.5), Math.tanh(-1), Math.tanh(-0.75), Math.tanh(2)])
    })

    it('should back propagate', function () {
        const input = new Volume([[-2, 1, -3, 4], [0.5, -1, -0.75, 2]])
        const outputGradients = [[-0.2, 0.7, 1.4, -5.2], [-1, 2.3, 1.5, -0.45]]
        const layer = new TanhLayer({inputShape: input.getShape()})

        layer.feedForward(input)
        layer.output.gradients[0][0] = outputGradients[0]
        layer.output.gradients[1][0] = outputGradients[1]
        layer.backPropagate()

        assert.deepEqual(input.gradients[0][0], [
            (1 - Math.pow(Math.tanh(-2), 2)) * outputGradients[0][0],
            (1 - Math.pow(Math.tanh(1), 2)) * outputGradients[0][1],
            (1 - Math.pow(Math.tanh(-3), 2)) * outputGradients[0][2],
            (1 - Math.pow(Math.tanh(4), 2)) * outputGradients[0][3]
        ])
        assert.deepEqual(input.gradients[1][0], [
            (1 - Math.pow(Math.tanh(0.5), 2)) * outputGradients[1][0],
            (1 - Math.pow(Math.tanh(-1), 2)) * outputGradients[1][1],
            (1 - Math.pow(Math.tanh(-0.75), 2)) * outputGradients[1][2],
            (1 - Math.pow(Math.tanh(2), 2)) * outputGradients[1][3]
        ])
    })
})

describe('Neural Network', function () {
    it('should build a classification network', function () {
        const nn = new NeuralNetwork()
        let input = []

        for (let i = 0; i < 4; i++) {
            input.push([])

            for (let j = 0; j < 160; j++) {
                input[i].push(Math.random())
            }
        }

        input = new Volume(input)

        nn.addLayer('conv1d', {nbKernels: 5, kernelSize: 3, stride: 1, inputShape: input.getShape()})
        nn.addLayer('maxpool1d', {kernelSize: 2, stride: 2})
        nn.addLayer('relu')
        nn.addLayer('fc', {nbNeurons: 20})
        nn.addLayer('softmax', {nbClasses: 10})

        const output = nn.predict(input)

        assert.equal(output.width, 10)

        const layers = []

        for (let i = 0; i < nn.layers.length; i++) {
            layers.push(nn.layers[i].constructor.name)
        }

        assert.deepEqual(layers, ['Convolution1DLayer', 'MaxPool1DLayer', 'ReluLayer', 'FullyConnectedLayer', 'FullyConnectedLayer', 'SoftmaxLayer'])
    })

    it('should build a regression network', function () {
        const nn = new NeuralNetwork()
        let input = []

        for (let i = 0; i < 4; i++) {
            input.push([])

            for (let j = 0; j < 160; j++) {
                input[i].push(Math.random())
            }
        }

        input = new Volume(input)

        nn.addLayer('conv1d', {nbKernels: 5, kernelSize: 3, stride: 1, inputShape: input.getShape()})
        nn.addLayer('maxpool1d', {kernelSize: 2, stride: 2})
        nn.addLayer('relu')
        nn.addLayer('fc', {nbNeurons: 20})
        nn.addLayer('regression', {nbOutputs: 5})

        const output = nn.predict(input)

        assert.equal(output.width, 5)

        const layers = []

        for (let i = 0; i < nn.layers.length; i++) {
            layers.push(nn.layers[i].constructor.name)
        }

        assert.deepEqual(layers, ['Convolution1DLayer', 'MaxPool1DLayer', 'ReluLayer', 'FullyConnectedLayer', 'FullyConnectedLayer', 'RegressionLayer'])
    })

    it('should add an activation function', function () {
        const nn = new NeuralNetwork()

        nn.addLayer('conv1d', {nbKernels: 5, kernelSize: 3, stride: 1, inputShape: [160, 1, 4], activation: 'relu'})

        const layers = []

        for (let i = 0; i < nn.layers.length; i++) {
            layers.push(nn.layers[i].constructor.name)
        }

        assert.deepEqual(layers, ['Convolution1DLayer', 'ReluLayer'])
    })

    it('should error on unknown activation function', function () {
        const nn = new NeuralNetwork()

        assert.throws(function () {
            nn.addLayer('conv1d', {
                nbKernels: 5,
                kernelSize: 3,
                stride: 1,
                inputShape: [4, 160],
                activation: 'foobar'
            })
        }, Error, 'Unknown activation function "foobar". Available functions : relu, sigmoid, tanh')
    })
})

describe('Volume', function () {
    it('should take 1D data', function () {
        const data = [1, 2, 3, 4]
        const volume = new Volume(data)

        assert.equal(volume.depth, 1)
        assert.equal(volume.height, 1)
        assert.equal(volume.width, 4)
        assert.deepEqual(volume.data.map(col => col.map(row => Array.from(row))), [[data]])
        assert.deepEqual(volume.gradients.map(col => col.map(row => Array.from(row))), [[data.map(() => 0)]])
    })

    it('should take 2D data', function () {
        const data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        const volume = new Volume(data)

        assert.equal(volume.depth, 2)
        assert.equal(volume.height, 1)
        assert.equal(volume.width, 4)
        assert.deepEqual(volume.data, data.map(row => [Float64Array.from(row)]))
        assert.deepEqual(volume.gradients, data.map((row) => [row.map(() => 0)]))
    })

    it('should take 3D data', function () {
        const data = [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
        ]
        const volume = new Volume(data)

        assert.equal(volume.depth, 2)
        assert.equal(volume.height, 3)
        assert.equal(volume.width, 4)
        assert.deepEqual(volume.data, data.map((col) => col.map((row) => Float64Array.from(row))))
        assert.deepEqual(volume.gradients, data.map((col) => col.map((row) => row.map(() => 0))))
    })

    it('should generate 3D data', function () {
        const width = 24
        const height = 12
        const depth = 4
        const volume = new Volume(width, height, depth)

        assert.equal(volume.depth, depth)
        assert.equal(volume.height, height)
        assert.equal(volume.width, width)
        assert.equal(volume.data.length, depth)
        assert.equal(volume.data[0].length, height)
        assert.equal(volume.data[0][0].length, width)
        assert.equal(volume.gradients.length, depth)
        assert.equal(volume.gradients[0].length, height)
        assert.equal(volume.gradients[0][0].length, width)
    })

    it('should generate 3D data with default value', function () {
        const width = 24
        const height = 12
        const depth = 4
        const defaultValue = 2.0
        const volume = new Volume(width, height, depth, defaultValue)

        for (let d = 0; d < depth; d++) {
            for (let h = 0; h < height; h++) {
                for (let w = 0; w < width; w++) {
                    assert.equal(volume.data[d][h][w], defaultValue)
                }
            }
        }
    })
})
