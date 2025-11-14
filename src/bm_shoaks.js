import Volume from './cnn/Volume.js'
import NeuralNetwork from './cnn/NeuralNetwork.js'
import _ from 'lodash'
import benchmark from 'benchmark'

const Benchmark = benchmark.runInContext({ _ })
const input = new Volume( 180, 1, 4)

const convNet = new NeuralNetwork()
convNet.addLayer('conv1d', {inputShape: input.getShape(), kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
convNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
convNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
convNet.addLayer('fc', {nbNeurons: 12, activation: 'relu'})
convNet.addLayer('regression', {nbOutputs: 2})

const pooledNet = new NeuralNetwork()
pooledNet.addLayer('conv1d', {inputShape: input.getShape(), kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
pooledNet.addLayer('maxpool1d', {kernelSize: 2, stride: 2})
pooledNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
pooledNet.addLayer('maxpool1d', {kernelSize: 3, stride: 2})
pooledNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
pooledNet.addLayer('fc', {nbNeurons: 12, activation: 'relu'})
convNet.addLayer('regression', {nbOutputs: 2})

const pooledMurkyNet = new NeuralNetwork()
pooledMurkyNet.addLayer('maxpool1d', {inputShape: input.getShape(), kernelSize: 2, stride: 2})
pooledMurkyNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
pooledMurkyNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
pooledMurkyNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1, activation: 'relu'})
pooledMurkyNet.addLayer('fc', {nbNeurons: 12, activation: 'relu'})
convNet.addLayer('regression', {nbOutputs: 2})

const suite = Benchmark.Suite('Convnet vs Poolnet')
suite.add('Conv Neural Net', () => {
    convNet.predict(input)
}).add('Conv + Pool Neural Net', () => {
    pooledNet.predict(input)
}).add('Pool + Conv Neural Net', () => {
    pooledMurkyNet.predict(input)
}).on('cycle', (event) => {
    console.log(String(event.target))
}).on('complete', function() {
    console.log('Fastest forward pass is ' + this.filter('fastest').map('name'))
    console.log('Average time ' + (this.filter('fastest')[0].stats.mean * 1000.0).toFixed(2) + 'ms')
}).run({async: true})
