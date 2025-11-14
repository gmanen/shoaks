const Volume = require('./Volume.js')
const Layer = require('./Layer.js')

class ReluLayer extends Layer {
    constructor(params) {
        super()

        this.inputShape = params.inputShape
        this.outputShape = params.outputShape || this.getOutputShape()
        this.output = new Volume(...this.outputShape, 0.0)
    }

    feedForward(input) {
        this.input = input
        this.output.zeroValues()

        for (let d = 0; d < this.output.depth; d++) {
            for (let h = 0; h < this.output.height; h++) {
                for (let w = 0; w < this.output.width; w++) {
                    if (input.data[d][h][w] > 0) {
                        this.output.data[d][h][w] = input.data[d][h][w]
                    }
                }
            }
        }

        return this.output
    }

    backPropagate() {
        for (let d = 0; d < this.output.depth; d++) {
            for (let h = 0; h < this.output.height; h++) {
                for (let w = 0; w < this.output.width; w++) {
                    if (this.output.data[d][h][w] <= 0) {
                        this.input.gradients[d][h][w] = 0.0
                    } else {
                        this.input.gradients[d][h][w] = this.output.gradients[d][h][w]
                    }
                }
            }
        }
    }

    getOutputShape() {
        return this.inputShape
    }

    clone() {
        return new ReluLayer({
            inputShape: this.inputShape
        })
    }
}

module.exports = ReluLayer
