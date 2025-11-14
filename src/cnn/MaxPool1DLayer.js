const Volume = require('./Volume.js')
const Layer = require('./Layer.js')

class MaxPool1DLayer extends Layer {
    constructor(params) {
        super()

        this.kernelSize = params.kernelSize
        this.inputShape = params.inputShape
        this.stride = params.stride
        this.outputShape = params.outputShape || this.getOutputShape()

        for (const length of this.outputShape) {
            if (!Number.isInteger(length)) {
                throw new Error("Output shape must be integers, " + JSON.stringify(this.outputShape) + " calculated.")
            }
        }

        this.output = new Volume(...this.outputShape, 0.0)
    }

    feedForward(input) {
        this.input = input

        for (let inputDepth = 0; inputDepth < input.depth; inputDepth++) {
            let outputIndex = 0

            for (let inputWidthOffset = 0; inputWidthOffset + this.kernelSize <= input.width; inputWidthOffset += this.stride) {
                let maxActivation = input.data[inputDepth][0][inputWidthOffset]

                for (let k = inputWidthOffset + 1; k < inputWidthOffset + this.kernelSize; k++) {
                    if (input.data[inputDepth][0][k] > maxActivation) {
                        maxActivation = input.data[inputDepth][0][k]
                    }
                }

                this.output.set(outputIndex, 0, inputDepth, maxActivation)
                outputIndex++
            }
        }

        return this.output
    }

    backPropagate() {
        this.input.zeroGradients()

        for (let d = 0; d < this.output.depth; d++) {
            for (let w = 0; w < this.output.width; w++) {
                let found = false

                for (let k = 0; k < this.kernelSize; k++) {
                    if (!found && this.input.data[d][0][w * this.stride + k] === this.output.data[d][0][w]) {
                        this.input.gradients[d][0][w * this.stride + k] += this.output.gradients[d][0][w]
                        found = true
                    }
                }
            }
        }
    }

    getOutputShape() {
        if (this.outputShape) {
            return this.outputShape
        }

        return [(this.inputShape[0] - this.kernelSize) / this.stride + 1, 1, this.inputShape[2]]
    }

    clone() {
        return new MaxPool1DLayer({
            kernelSize: this.kernelSize,
            inputShape: this.inputShape,
            stride: this.stride,
            outputShape: this.outputShape
        })
    }
}

module.exports = MaxPool1DLayer
