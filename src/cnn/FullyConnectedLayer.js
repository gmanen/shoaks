import Volume from './Volume.js'
import Layer from './Layer.js'
import {flatten} from './utils.js'

export default class FullyConnectedLayer extends Layer {
    constructor(params) {
        super(params.weights || [], params.biases)

        this.nbNeurons = params.nbNeurons
        this.inputShape = params.inputShape
        this.outputShape = params.outputShape || this.getOutputShape()

        this.inputShape = [this.inputShape[0], this.inputShape[1] || 1, this.inputShape[2] || 1]
        this.inputWidth = this.inputShape[0]
        this.inputHeight = this.inputShape[1]
        this.inputDepth = this.inputShape[2]
        this.inputSize = this.inputWidth * this.inputHeight * this.inputDepth

        if (this.weights.length === 0) {
            for (let i = 0; i < this.nbNeurons; i++) {
                this.weights.push(new Volume(this.inputSize))
            }
        }

        if (typeof this.biases === "undefined") {
            this.biases = new Volume(this.nbNeurons, 1, 1, 0.0)
        }

        this.output = new Volume(...this.outputShape, 0.0)
        this.flattenedInput = new Float64Array(this.inputSize)
    }

    feedForward(input) {
        this.input = input

        this.inputVolume = input instanceof Volume ? input : null

        const flattenedInput = this.getFlattenedInput()

        for (let n = 0; n < this.nbNeurons; n++) {
            const weights = this.weights[n].data[0][0]
            let sum = this.biases.data[0][0][n]

            for (let i = 0; i < flattenedInput.length; i++) {
                sum += flattenedInput[i] * weights[i]
            }

            this.output.data[0][0][n] = sum
        }

        return this.output
    }

    backPropagate() {
        if (this.inputVolume) {
            this.inputVolume.zeroGradients()
        }

        const flattenedInput = this.getFlattenedInput()
        const weightWidth = this.inputWidth
        const weightArea = weightWidth * this.inputHeight

        for (let n = 0; n < this.nbNeurons; n++) {
            const grad = this.output.gradients[0][0][n]
            const weights = this.weights[n].data[0][0]
            const weightGradients = this.weights[n].gradients[0][0]

            for (let i = 0; i < flattenedInput.length; i++) {
                if (this.inputVolume) {
                    const d = Math.floor(i / weightArea)
                    const h = Math.floor((i - d * weightArea) / weightWidth)
                    const w = i - d * weightArea - h * weightWidth

                    this.inputVolume.gradients[d][h][w] += weights[i] * grad
                }

                weightGradients[i] += flattenedInput[i] * grad
            }

            this.biases.gradients[0][0][n] += grad
        }
    }

    getFlattenedInput() {
        if (this.input instanceof Float64Array && this.input.length === this.inputSize) {
            return this.input
        }

        if (this.flattenedInput.length !== this.inputSize) {
            this.flattenedInput = new Float64Array(this.inputSize)
        }

        return flatten(this.input, this.flattenedInput)
    }

    getOutputShape() {
        return [this.nbNeurons, 1, 1]
    }

    clone() {
        return new FullyConnectedLayer({
            nbNeurons: this.nbNeurons,
            inputShape: this.inputShape,
            weights: this.weights.map(weight => new Volume(weight.data)),
            biases: new Volume(this.biases.data)
        })
    }
}
