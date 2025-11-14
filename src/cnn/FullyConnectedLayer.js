const Volume = require('./Volume.js')
const Layer = require('./Layer.js')

class FullyConnectedLayer extends Layer {
    constructor(params) {
        super(params.weights || [], params.biases)

        this.nbNeurons = params.nbNeurons
        this.inputShape = params.inputShape
        this.outputShape = params.outputShape || this.getOutputShape()

        if (this.weights.length === 0) {
            const nbInputs = this.inputShape[0] * this.inputShape[1] * this.inputShape[2]
            for (let i = 0; i < this.nbNeurons; i++) {
                this.weights.push(new Volume(nbInputs))
            }
        }

        if (typeof this.biases === "undefined") {
            this.biases = new Volume(this.nbNeurons, 1, 1, 0.0)
        }

        this.output = new Volume(...this.outputShape, 0.0)
    }

    feedForward(input) {
        this.input = input

        for (let n = 0; n < this.nbNeurons; n++) {
            let weightIndex = 0

            this.output.data[0][0][n] = this.biases.data[0][0][n]

            for (let d = 0; d < input.depth; d++) {
                for (let h = 0; h < input.height; h++) {
                    for (let w = 0; w < input.width; w++) {
                        this.output.data[0][0][n] += input.data[d][h][w] * this.weights[n].data[0][0][weightIndex]
                        weightIndex++
                    }
                }
            }
        }

        return this.output
    }

    backPropagate() {
        this.input.zeroGradients()

        for (let n = 0; n < this.nbNeurons; n++) {
            let weightIndex = 0

            for (let d = 0; d < this.input.depth; d++) {
                for (let h = 0; h < this.input.height; h++) {
                    for (let w = 0; w < this.input.width; w++) {
                        this.input.gradients[d][h][w] += this.weights[n].data[0][0][weightIndex] * this.output.gradients[0][0][n]
                        this.weights[n].gradients[0][0][weightIndex] += this.input.data[d][h][w] * this.output.gradients[0][0][n]
                        weightIndex++
                    }
                }
            }

            this.biases.gradients[0][0][n] += this.output.gradients[0][0][n]
        }
    }

    getOutputShape() {
        return [this.nbNeurons, 1, 1]
    }

    clone() {
        return new FullyConnectedLayer({
            nbNeurons: this.nbNeurons,
            inputShape: this.inputShape,
            weights: this.weights,
            biases: this.biases
        })
    }
}

module.exports = FullyConnectedLayer
