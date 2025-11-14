let return_v = false
let v_val = 0.0
let randomGaussianAndrej = function() {
    if(return_v) {
        return_v = false
        return v_val
    }

    const u = 2*Math.random()-1
    const v = 2*Math.random()-1
    const r = u*u + v*v

    if(r === 0 || r > 1) {
        return randomGaussianAndrej()
    }

    const c = Math.sqrt(-2*Math.log(r)/r)

    v_val = v*c
    return_v = true

    return u*c
}

function randomGaussian(mean, deviation) {
    mean = mean || 0
    deviation = deviation || 1

    return mean + randomGaussianAndrej() * deviation
}

function dotProduct(weights, input) {
    const output = []

    for (let i = 0; i < weights.length; i++) {
        let weightedSum = 0

        for (let j = 0; j < input.length; j++) {
            weightedSum += weights[i][j] * input[j]
        }

        output.push(weightedSum)
    }

    return output
}

function addVector(alice, bob) {
    const output = []

    for (let i = 0; i < alice.length; i++) {
        output.push(alice[i] + bob[i])
    }

    return output
}

function sigmoid(value) {
    return 1 / (1 + Math.exp(-value))
}

function flatten(array) {
    return array.reduce(
        (flattened, current) => flattened.concat(Array.isArray(current) ? flatten(current) : current),
        []
    )
}

function zeros(length) {
    return new Float64Array(length);
}

exports.randomGaussian = randomGaussian
exports.dotProduct = dotProduct
exports.addVector = addVector
exports.sigmoid = sigmoid
exports.flatten = flatten
exports.zeros = zeros
