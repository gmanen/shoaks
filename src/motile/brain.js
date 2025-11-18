import NeuralNetwork from '../cnn/NeuralNetwork.js'
import Volume from '../cnn/Volume.js'
import {flatten} from '../cnn/utils.js'

export default class Brain {
    constructor(sightInputs, additionalInputs = 0, outputs) {
        this.flattenBuffer = []

        if (sightInputs !== undefined) {
            this.convNet = new NeuralNetwork()
            this.convNet.addLayer('conv1d', {inputShape: [sightInputs, 1, 1], kernelSize: 3, nbKernels: 10, stride: 1})
            this.convNet.addLayer('relu')
            this.convNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1})
            this.convNet.addLayer('relu')
            this.convNet.addLayer('conv1d', {kernelSize: 3, nbKernels: 10, stride: 1})
            this.convNet.addLayer('relu')

            this.additionalInputs = additionalInputs
            this.flattenedConvSize = this.convNet.getOutputShape().reduce((total, value) => value * total, 1)
            this.flattenBuffer = new Float64Array(this.flattenedConvSize)

            this.denseNet = new NeuralNetwork()
            this.denseNet.addLayer('fc', {inputShape: [this.flattenedConvSize + additionalInputs, 1, 1], nbNeurons: 12})
            this.denseNet.addLayer('relu')
            this.denseNet.addLayer('fc', {nbNeurons: outputs})

            this.denseInput = new Float64Array(this.flattenedConvSize + additionalInputs)
        }
    }

    evaluate(inputArray, additionalInputs = []) {
        const extras = Array.isArray(additionalInputs) ? additionalInputs : [additionalInputs]
        if (this.flattenBuffer.length !== this.flattenedConvSize) {
            this.flattenBuffer = new Float64Array(this.flattenedConvSize)
        }

        const denseInputSize = this.flattenedConvSize + extras.length

        if (!this.denseInput || this.denseInput.length !== denseInputSize) {
            this.denseInput = new Float64Array(denseInputSize)
        }

        flatten(this.convNet.predict(inputArray), this.flattenBuffer)
        this.denseInput.set(this.flattenBuffer)

        for (let i = 0; i < extras.length; i++) {
            this.denseInput[this.flattenedConvSize + i] = extras[i]
        }

        return this.denseNet.predict(this.denseInput)
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
            this.mutateLayerWeights(layer, rate)
        }
    }

    randomize() {
        this.randomizeNeuralNet(this.convNet)
        this.randomizeNeuralNet(this.denseNet)
    }

    randomizeNeuralNet(nn) {
        for (const layer of nn.layers) {
            this.randomizeLayerWeights(layer)
        }
    }

    mutateLayerWeights(layer, rate) {
        if (!layer.weights || !layer.biases) {
            return
        }

        const weightVolumes = Array.isArray(layer.weights) ? layer.weights : [layer.weights]

        for (const weightVolume of weightVolumes) {
            this.mutateVolume(weightVolume, rate)
        }

        this.mutateVolume(layer.biases, rate)
    }

    randomizeLayerWeights(layer) {
        if (!layer.weights || !layer.biases) {
            return
        }

        const weightVolumes = Array.isArray(layer.weights) ? layer.weights : [layer.weights]

        for (let weightIndex = 0; weightIndex < weightVolumes.length; weightIndex++) {
            const volume = weightVolumes[weightIndex]
            layer.weights[weightIndex] = new Volume(volume.width, volume.height, volume.depth)
        }

        layer.biases = new Volume(layer.biases.width, layer.biases.height, layer.biases.depth)
    }

    mutateVolume(volume, rate) {
        const sqrtNbWeights = Math.sqrt(volume.width * volume.height * volume.depth)

        for (let d = 0; d < volume.depth; d++) {
            for (let h = 0; h < volume.height; h++) {
                for (let w = 0; w < volume.width; w++) {
                    if (p.random() < rate) {
                        volume.data[d][h][w] += p.randomGaussian(0, 1 / sqrtNbWeights)
                    }
                }
            }
        }
    }
}
