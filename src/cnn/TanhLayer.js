const Volume = require('./Volume.js')
const Layer = require('./Layer.js')

class TanhLayer extends Layer {
    constructor(params) {
        super()

        this.inputShape = params.inputShape
        this.outputShape = params.outputShape || this.getOutputShape()
        this.output = new Volume(...this.outputShape, 0.0)
    }

    feedForward(input) {
        this.input = input

        for (let d = 0; d < this.output.depth; d++) {
            for (let h = 0; h < this.output.height; h++) {
                for (let w = 0; w < this.output.width; w++) {
                    this.output.data[d][h][w] = Math.tanh(input.data[d][h][w])
                }
            }
        }

        return this.output
    }

    backPropagate() {
        for (let d = 0; d < this.output.depth; d++) {
            for (let h = 0; h < this.output.height; h++) {
                for (let w = 0; w < this.output.width; w++) {
                    this.input.gradients[d][h][w] = (1 - this.output.data[d][h][w] * this.output.data[d][h][w]) * this.output.gradients[d][h][w]
                }
            }
        }
    }

    getOutputShape() {
        return this.inputShape
    }

    clone() {
        return new TanhLayer({
            inputShape: this.inputShape,
        })
    }
}

module.exports = TanhLayer
