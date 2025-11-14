const Volume = require('./Volume.js')
const Layer = require('./Layer.js')

class Convolution1DLayer extends Layer {
    constructor(params) {
        super(params.weights || [], params.biases)

        this.nbKernels = params.nbKernels
        this.kernelSize = params.kernelSize
        this.inputShape = params.inputShape
        this.stride = params.stride
        this.padding = params.padding || 0
        this.conserveShape = params.conserveShape || false

        if (this.inputShape[1] !== 1) {
            throw new Error("Conv1D Layer can only take input in one dimension (width). Input height should be 1, " + this.inputShape[1] + " given.")
        }

        if (this.padding === 0 && this.conserveShape) {
            this.padding = (this.kernelSize - 1) / 2
        }

        this.outputShape = params.outputShape || this.getOutputShape()

        for (const length of this.outputShape) {
            if (!Number.isInteger(length)) {
                throw new Error("Output shape must be integers, " + JSON.stringify(this.outputShape) + " calculated.")
            }
        }

        if (this.weights.length === 0) {
            for (let i = 0; i < this.nbKernels; i++) {
                this.weights.push(new Volume(this.kernelSize, 1, this.inputShape[2]))
            }
        }

        if (typeof this.biases === "undefined") {
            this.biases = new Volume(this.nbKernels, 1, 1, 0.0)
        }

        this.output = new Volume(...this.outputShape, 0.0)
    }

    feedForward(input) {
        this.input = input

        for (let step = 0, offset = -this.padding; offset + this.kernelSize <= input.width + this.padding; step++, offset += this.stride) {
            for (let kernelIndex = 0; kernelIndex < this.nbKernels; kernelIndex++) {
                this.output.data[kernelIndex][0][step] = this.biases.data[0][0][kernelIndex]

                for (let kernelSizeIndex = 0; kernelSizeIndex < this.kernelSize; kernelSizeIndex++) {
                    let inputWidthOffset = offset + kernelSizeIndex

                    for (let inputDepthIndex = 0; inputDepthIndex < input.depth; inputDepthIndex++) {
                        if (inputWidthOffset >= 0 && inputWidthOffset < input.width) {
                            this.output.data[kernelIndex][0][step] += input.data[inputDepthIndex][0][inputWidthOffset] * this.weights[kernelIndex].data[inputDepthIndex][0][kernelSizeIndex]
                        }
                    }
                }
            }
        }

        return this.output
    }

    backPropagate() {
        this.input.zeroGradients()

        for (let outputDepth = 0; outputDepth < this.output.depth; outputDepth++) {
            for (let outputWidth = 0; outputWidth < this.output.width; outputWidth++) {
                for (let inputDepth = 0; inputDepth < this.input.depth; inputDepth++) {
                    const offset = outputWidth * this.stride - this.padding

                    for (let k = 0; k < this.kernelSize; k++) {
                        if (offset + k >= 0 && offset + k < this.input.width) {
                            this.input.gradients[inputDepth][0][offset + k] += this.weights[outputDepth].data[inputDepth][0][k] * this.output.gradients[outputDepth][0][outputWidth]
                            this.weights[outputDepth].gradients[inputDepth][0][k] += this.input.data[inputDepth][0][offset + k] * this.output.gradients[outputDepth][0][outputWidth]
                        }
                    }
                }

                this.biases.gradients[0][0][outputDepth] += this.output.gradients[outputDepth][0][outputWidth]
            }
        }
    }

    getOutputShape() {
        if (this.outputShape) {
            return this.outputShape
        }

        return [(this.inputShape[0] - this.kernelSize + 2 * this.padding) / this.stride + 1, 1, this.nbKernels]
    }

    clone() {
        return new Convolution1DLayer({
            nbKernels: this.nbKernels,
            kernelSize: this.kernelSize,
            inputShape: this.inputShape,
            stride: this.stride,
            outputShape: this.outputShape,
            weights: this.weights,
            biases: this.biases
        })
    }
}

module.exports = Convolution1DLayer
