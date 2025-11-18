import NeuralNetwork from '../cnn/NeuralNetwork.js'
import {flatten} from '../cnn/utils.js'

export default class Brain {
    constructor(sightInputs, additionalInputs = 0, outputs) {
        if (sightInputs !== undefined) {
            this.convNet = new NeuralNetwork()
            this.convNet.addLayer('conv1d', {inputShape: [sightInputs, 1, 1], kernelSize: 3, nbKernels: 10, stride: 1})
            this.convNet.addLayer('relu')
            this.convNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1})
            this.convNet.addLayer('relu')
            this.convNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1})
            this.convNet.addLayer('relu')

            this.denseNet = new NeuralNetwork()
            this.denseNet.addLayer('fc', {inputShape: [this.convNet.getOutputShape().reduce((total, value) => value * total, 1) + additionalInputs], nbNeurons: 12})
            this.denseNet.addLayer('relu')
            this.denseNet.addLayer('fc', {nbNeurons: outputs})
        }
    }

    evaluate(inputArray, additionalInputs = []) {
        const extras = Array.isArray(additionalInputs) ? additionalInputs : [additionalInputs]

        return this.denseNet.predict(flatten(this.convNet.predict(inputArray)).concat(extras))
    }

    clone() {
        const brain = new Brain()
        brain.convNet = this.convNet.clone()
        brain.denseNet = this.denseNet.clone()

        return brain
    }

    crossover(parentBrain) {
        const childBrain = this.clone()

        this.crossoverNeuralNet(this.convNet, parentBrain.convNet)
        this.crossoverNeuralNet(this.denseNet, parentBrain.denseNet)

        return childBrain
    }

    crossoverNeuralNet(myNN, parentNN) {
        for (let layerIndex = 0; layerIndex < myNN.layers.length; layerIndex++) {
            for (let i = 0; i < myNN.layers[layerIndex].biases.length; i++) {
                if (p.random() < 0.5) { // Picks the other parent
                    myNN.layers[layerIndex].biases[i] = parentNN.layers[layerIndex].biases[i]
                    myNN.layers[layerIndex].weights[i] = parentNN.layers[layerIndex].weights[i]
                }
            }
        }
    }

    mutate(rate) {
        this.mutateNeuralNet(this.convNet, rate)
        this.mutateNeuralNet(this.denseNet, rate)
    }

    mutateNeuralNet(nn, rate) {
        for (const layer of nn.layers) {
            for (let i = 0; i < layer.biases.length; i++) {
                if (p.random() < rate) {
                    let sqrtNbWeights = layer.weights[i].length

                    if (Array.isArray(layer.weights[i][0])) {
                        sqrtNbWeights *= layer.weights[i][0].length
                    }

                    sqrtNbWeights = Math.sqrt(sqrtNbWeights)

                    const mutateMapper = (value) => Array.isArray(value)
                        ? value.map(mutateMapper)
                        : value + p.randomGaussian(0, 1 / sqrtNbWeights)

                    layer.weights[i] = layer.weights[i].map(mutateMapper)
                }
            }
        }
    }
}
