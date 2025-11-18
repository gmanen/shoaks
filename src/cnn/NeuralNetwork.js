import Convolution1DLayer from './Convolution1DLayer.js'
import MaxPool1DLayer from './MaxPool1DLayer.js'
import FullyConnectedLayer from './FullyConnectedLayer.js'
import ReluLayer from './ReluLayer.js'
import SigmoidLayer from './SigmoidLayer.js'
import TanhLayer from './TanhLayer.js'
import SoftmaxLayer from './SoftmaxLayer.js'
import RegressionLayer from './RegressionLayer.js'

export default class NeuralNetwork {
    constructor() {
        this.layers = []
    }

    addLayer(type, params = {}) {
        let layer
        let activationFunction
        const currentLayersLength = this.layers.length

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
                throw new Error("param 'nbClasses' is required for Softmax layer")
            }

            params.nbNeurons = params.nbClasses
            delete params.nbClasses

            const fcLayer = new FullyConnectedLayer(params)

            this.layers.push(fcLayer)

            layer = new SoftmaxLayer({inputShape: fcLayer.getOutputShape()})
        } else if ('regression' === type) {
            if (!params.hasOwnProperty('nbOutputs')) {
                throw new Error("param 'nbClasses' is required for Regression layer")
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

        return nn
    }
}
