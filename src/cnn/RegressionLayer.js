const Volume = require('./Volume.js')
const Layer = require('./Layer.js')

class RegressionLayer extends Layer {
    constructor(params) {
        super()

        if (params.inputShape[1] !== 1 || params.inputShape[2] !== 1) {
            throw "Regression only accepts one dimensional arrays as input, use a FC Layer first.";
        }

        this.inputShape = params.inputShape
        this.outputShape = params.outputShape || this.getOutputShape()
        this.output = new Volume(...this.outputShape, 0.0)
    }

    feedForward(input) {
        this.input = input

        for (let i = 0; i < input.width; i++) {
            this.output.data[0][0][i] = input.data[0][0][i]
        }

        return this.output
    }

    backPropagate(expectedValues) {
        let loss = 0.0

        for(let i = 0; i < this.output.width; i++) {
            const costDerivative = this.input.data[0][0][i] - expectedValues[i]

            this.input.gradients[0][0][i] = costDerivative
            loss += 0.5 * costDerivative * costDerivative
        }

        return loss
    }

    getOutputShape() {
        return this.inputShape
    }

    clone() {
        return new RegressionLayer({
            inputShape: this.inputShape
        })
    }
}

module.exports = RegressionLayer
