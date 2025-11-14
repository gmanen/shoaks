const Convolution1DLayer = require('./Convolution1DLayer')
const MaxPool1DLayer = require('./MaxPool1DLayer')
const FullyConnectedLayer = require('./FullyConnectedLayer')
const ReluLayer = require('./ReluLayer')
const SigmoidLayer = require('./SigmoidLayer')
const TanhLayer = require('./TanhLayer')
const SoftmaxLayer = require('./SoftmaxLayer')
const RegressionLayer = require('./RegressionLayer')

class NeuralNetwork {
    constructor() {
        this.layers = []
    }

    addLayer(type, params) {
        let layer, activationFunction
        const currentLayersLength = this.layers.length
        params = params || {}

        if (currentLayersLength === 0 && !params.hasOwnProperty('inputShape')) {
            throw new Error('First layer must have explicit input shape')
        }

        if (!params.hasOwnProperty('inputShape')) {
            params.inputShape = this.layers[currentLayersLength - 1].getOutputShape()
        }

        if (params.hasOwnProperty('activation')) {
            if (['relu', 'sigmoid', 'tanh'].indexOf(params.activation) < 0) {
                throw new Error('Unknown activation function "' + params.activation + '". Available functions : ' + ['relu', 'sigmoid', 'tanh'].join(', '))
            }

            activationFunction = params.activation
            delete params.activation
        }

        if ('conv1d' === type) {
            layer = new Convolution1DLayer(params)
        } else if ('maxpool1d' === type) {
            layer = new MaxPool1DLayer(params)
        } else if ('fc' === type) {
            layer = new FullyConnectedLayer(params)
        } else if ('relu' === type) {
            layer = new ReluLayer(params)
        } else if ('sigmoid' === type) {
            layer = new SigmoidLayer(params)
        } else if ('tanh' === type) {
            layer = new TanhLayer(params)
        } else if ('softmax' === type) {
            if (!params.hasOwnProperty('nbClasses')) {
                throw "param 'nbClasses' is required for Softmax layer"
            }

            params.nbNeurons = params.nbClasses
            delete params.nbClasses

            const fcLayer = new FullyConnectedLayer(params)

            this.layers.push(fcLayer)

            layer = new SoftmaxLayer({inputShape: fcLayer.getOutputShape()})
        } else if ('regression' === type) {
            if (!params.hasOwnProperty('nbOutputs')) {
                throw "param 'nbClasses' is required for Regression layer"
            }

            params.nbNeurons = params.nbOutputs
            delete params.nbOutputs

            const fcLayer = new FullyConnectedLayer(params)

            this.layers.push(fcLayer)

            layer = new RegressionLayer({inputShape: fcLayer.getOutputShape()})
        }

        this.layers.push(layer)

        if (activationFunction) {
            this.addLayer(activationFunction)
        }
    }

    getOutputShape()
    {
        return this.layers[this.layers.length - 1].getOutputShape()
    }

    predict(input) {
        for (const layer of this.layers) {
            input = layer.feedForward(input)
        }

        return input
    }

    clone() {
        const nn = new NeuralNetwork()

        for (const layer of this.layers) {
            nn.layers.push(layer.clone())
        }
    }
}

module.exports = NeuralNetwork
