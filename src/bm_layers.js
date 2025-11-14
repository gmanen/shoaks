import Volume from './cnn/Volume.js'
import Convolution1DLayer from './cnn/Convolution1DLayer.js'
import MaxPool1DLayer from './cnn/MaxPool1DLayer.js'
import FullyConnectedLayer from './cnn/FullyConnectedLayer.js'
import ReluLayer from './cnn/ReluLayer.js'
import SigmoidLayer from './cnn/SigmoidLayer.js'
import TanhLayer from './cnn/TanhLayer.js'
import SoftmaxLayer from './cnn/SoftmaxLayer.js'
import RegressionLayer from './cnn/RegressionLayer.js'
import _ from 'lodash'
import benchmark from 'benchmark'

const Benchmark = benchmark.runInContext({ _ })

const input2d = new Volume( 180, 1, 4)
const inputFinal = new Volume( 1000, 1, 1)

const conv1D = new Convolution1DLayer({nbKernels: 5, kernelSize: 3, stride: 1, inputShape: input2d.getShape()})
const maxPool = new MaxPool1DLayer({kernelSize: 2, stride: 2, inputShape: input2d.getShape()})
const fc = new FullyConnectedLayer({nbNeurons: 12, inputShape: input2d.getShape()})
const relu = new ReluLayer({inputShape: input2d.getShape()})
const sigmoid = new SigmoidLayer({inputShape: input2d.getShape()})
const tanh = new TanhLayer({inputShape: input2d.getShape()})
const softmax = new SoftmaxLayer({inputShape: inputFinal.getShape()})
const regression = new RegressionLayer({inputShape: inputFinal.getShape()})

const conv1DGradients = new Volume( 178, 1, 5)
const maxPoolGradients = new Volume( 90, 1, 4)
const fcGradients = new Volume( 12, 1, 1)
const activationGradients = new Volume( ...input2d.getShape())
const randomClass = (Math.random() * inputFinal.getShape().reduce((total, val) => total * val, 1)) |0
const regressionExpectedOutput = (new Volume(...inputFinal.getShape())).data[0][0]

const suite = Benchmark.Suite('Feed forward on each layer type independently')
suite.add('Conv1D forward and back with '+JSON.stringify(input2d.getShape())+' ('+input2d.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    conv1D.feedForward(input2d)
    conv1D.output.gradients = conv1DGradients.data
    conv1D.backPropagate()
}).add('MaxPool forward and back with '+JSON.stringify(input2d.getShape())+' ('+input2d.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    maxPool.feedForward(input2d)
    maxPool.output.gradients = maxPoolGradients.data
    maxPool.backPropagate()
}).add('FC forward and back with '+JSON.stringify(input2d.getShape())+' ('+input2d.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    fc.feedForward(input2d)
    fc.output.gradients = fcGradients.data
    fc.backPropagate()
}).add('Relu forward and back with '+JSON.stringify(input2d.getShape())+' ('+input2d.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    relu.feedForward(input2d)
    relu.output.gradients = activationGradients.data
    relu.backPropagate()
}).add('Sigmoid forward and back with '+JSON.stringify(input2d.getShape())+' ('+input2d.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    sigmoid.feedForward(input2d)
    sigmoid.output.gradients = activationGradients.data
    sigmoid.backPropagate()
}).add('Tanh forward and back with '+JSON.stringify(input2d.getShape())+' ('+input2d.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    tanh.feedForward(input2d)
    tanh.output.gradients = activationGradients.data
    tanh.backPropagate()
}).add('Softmax forward and back with '+JSON.stringify(inputFinal.getShape())+' ('+inputFinal.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    softmax.feedForward(inputFinal)
    softmax.backPropagate(randomClass)
}).add('Regression forward and back with '+JSON.stringify(inputFinal.getShape())+' ('+inputFinal.getShape().reduce((total, val) => total * val, 1)+') inputs', () => {
    regression.feedForward(inputFinal)
    regression.backPropagate(regressionExpectedOutput)
}).on('cycle', (event) => {
    console.log(String(event.target))
    console.log('average time: ' + (event.target.stats.mean * 1000).toFixed(3) + 'ms')
}).on('error', (event) => {
    console.log('Suite error', event.target.error)
}).run({async: true})
