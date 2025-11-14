const {zeros, randomGaussian} = require('./utils')

class Volume {
    constructor(width, height, depth, defaultValue) {
        this.data = []
        this.gradients = []

        if('[object Array]' === Object.prototype.toString.call(width)) {
            if('[object Array]' === Object.prototype.toString.call(width[0])) {
                this.depth = width.length

                if(['[object Array]', '[object Float64Array]'].indexOf(Object.prototype.toString.call(width[0][0])) >= 0) {
                    this.height = width[0].length
                    this.width = width[0][0].length
                } else {
                    this.height = 1
                    this.width = width[0].length
                }
            } else {
                this.depth = 1
                this.height = 1
                this.width = width.length
            }

            let data = width

            if (this.height === 1 && this.depth === 1) {
                data = [[data]]
            } else if (this.height === 1 && this.depth > 1) {
                data = data.map(row => [row])
            }

            for (let d = 0; d < this.depth; d++) {
                this.data.push([])
                this.gradients.push([])

                for (let h = 0; h < this.height; h++) {
                    this.data[d].push(zeros(this.width))
                    this.gradients[d].push(zeros(this.width))

                    for (let w = 0; w < this.width; w++) {
                        this.data[d][h][w] = data[d][h][w]
                    }
                }
            }
        } else {
            this.width = width
            this.height = height || 1
            this.depth = depth || 1

            const scale = Math.sqrt(1.0 / (this.width * this.height * this.depth))
            const hasDefaultValue = typeof defaultValue !== "undefined"

            for (let d = 0; d < this.depth; d++) {
                this.data.push([])
                this.gradients.push([])

                for (let h = 0; h < this.height; h++) {
                    this.data[d].push(zeros(this.width))
                    this.gradients[d].push(zeros(this.width))

                    if (!hasDefaultValue || defaultValue != 0) {
                        for (let w = 0; w < this.width; w++) {
                            this.data[d][h][w] = hasDefaultValue ? defaultValue : randomGaussian(0, scale)
                        }
                    }
                }
            }
        }
    }

    getShape() {
        return [this.width, this.height, this.depth]
    }

    get(w, h, d) {
        return this.data[d][h][w]
    }

    set(w, h, d, value) {
        this.data[d][h][w] = value
    }

    increment(w, h, d, value) {
        this.data[d][h][w] = this.data[d][h][w] + value
    }

    zeroValues() {
        for (let d = 0; d < this.depth; d++) {
            for (let h = 0; h < this.height; h++) {
                for (let w = 0; w < this.width; w++) {
                    this.data[d][h][w] = 0.0
                }
            }
        }
    }

    zeroGradients() {
        for (let d = 0; d < this.depth; d++) {
            for (let h = 0; h < this.height; h++) {
                for (let w = 0; w < this.width; w++) {
                    this.gradients[d][h][w] = 0.0
                }
            }
        }
    }
}

module.exports = Volume
