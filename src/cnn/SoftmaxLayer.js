const Volume = require('./Volume.js')
const Layer = require('./Layer.js')

class SoftmaxLayer extends Layer {
    constructor(params) {
        super()

        if (params.inputShape[1] !== 1 || params.inputShape[2] !== 1) {
            throw "Softmax only accepts one dimensional arrays as input, use a FC Layer first.";
        }

        this.inputShape = params.inputShape
    }

    feedForward(input) {
        this.input = input
        this.output = new Volume(...this.getOutputShape(), 0.0)
        const maxInput = Math.max(...(input.data[0][0]))
        let exponentialSum = 0.0

        for (let i = 0; i < input.width; i++) {
            this.output.data[0][0][i] = Math.exp(input.data[0][0][i] - maxInput)
            exponentialSum += this.output.data[0][0][i]
        }

        for (let i = 0; i < input.width; i++) {
            this.output.data[0][0][i] /= exponentialSum
        }

        return this.output
    }

    backPropagate(expectedClass) {
        for(let i = 0; i < this.output.width; i++) {
            const expectedOutput = i === expectedClass ? 1.0 : 0.0
            this.input.gradients[0][0][i] = this.output.data[0][0][i] - expectedOutput
        }

        // loss is the class negative log likelihood
        return -Math.log(this.output.data[0][0][expectedClass]);
    }

    getOutputShape() {
        return this.inputShape
    }

    clone() {
        return new SoftmaxLayer({
            inputShape: this.inputShape,
        })
    }
}

module.exports = SoftmaxLayer
