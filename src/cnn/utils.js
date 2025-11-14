let returnV = false
let vVal = 0.0
const randomGaussianAndrej = () => {
    if (returnV) {
        returnV = false
        return vVal
    }

    const u = 2 * Math.random() - 1
    const v = 2 * Math.random() - 1
    const r = u * u + v * v

    if (r === 0 || r > 1) {
        return randomGaussianAndrej()
    }

    const c = Math.sqrt(-2 * Math.log(r) / r)

    vVal = v * c
    returnV = true

    return u * c
}

export const randomGaussian = (mean = 0, deviation = 1) => mean + randomGaussianAndrej() * deviation

export const dotProduct = (weights, input) => {
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

export const addVector = (alice, bob) => {
    const output = []

    for (let i = 0; i < alice.length; i++) {
        output.push(alice[i] + bob[i])
    }

    return output
}

export const sigmoid = (value) => 1 / (1 + Math.exp(-value))

export const flatten = (array) => array.reduce(
    (flattened, current) => flattened.concat(Array.isArray(current) ? flatten(current) : current),
    []
)

export const zeros = (length) => new Float64Array(length)
