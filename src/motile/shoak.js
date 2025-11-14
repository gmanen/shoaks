const {motileBehaviors} = require('./motile')
const {canSee} = require('./sighted')
const Brain = require('./brain')
const {topDownWidth, topDownHeight} = require('../config')
const {Circle} = require('../quadtree/quadtree')
let environment = require ('../environment')

const generateCircle = self => {
    return new Circle(self.position.x, self.position.y, self.radius)
}

const getShoakRadius = self => {
    return 5 + self.mass
}

const Shoak = (id, brain, shoakColor) => {
    const baseSpeed = 4
    const baseMass = 15
    const fov = parseInt(window.shoakFov)
    const resolution = parseFloat(window.shoakResolution)

    if (!brain) {
        const layers = []

        for (let i = 0; i < parseInt(window.shoakNNComplexity); i++) {
            layers.push(parseInt(window.shoakNNSize))
        }

        brain = new Brain(fov * resolution + 1, layers, 2, 'relu')
        brain.randomize()
    }

    const velocity = p5.Vector.random2D()
    velocity.setMag(baseSpeed)

    const self = {
        id: 'shoak-' + id,
        baseSpeed,
        minSpeed: 1,
        maxSpeed: 8,
        maxForce: 0.2,
        mass: baseMass,
        position: p.createVector(p.random(topDownWidth), p.random(topDownHeight)),
        velocity: velocity,
        acceleration: p.createVector(),
        fov,
        resolution, // Increment step size for the rays simulating the shark's vision
        sightRadius: parseInt(window.shoakSightRadius),
        maxMass: 30,
        brain,
        currentAge: 0,
        score: 0, // Useful mass eaten by the shark
        color: shoakColor || (360 + Math.floor(p.random(-30, 31))) % 360,
        sight: [], // Current sight is stored to be displayed
        generateShape: generateCircle,
        angles: [],
        angleSD: 0,
        neuralNetResults: []
    }

    self.shape = generateCircle(self)
    self.radius = getShoakRadius(self)

    const shoakBehaviors = self => ({
        think: (sketch) => {
            const sight = self.see(environment.qtree, ['foish'], self.sight, sketch)

            if (debug) {
                const velocity = p5.Vector.fromAngle(self.velocity.heading(), 100)
                sketch.stroke(0, 0, 255)
                sketch.strokeWeight(2)
                sketch.line(self.position.x, self.position.y, self.position.x + velocity.x, self.position.y + velocity.y)
            }

            const halfFov = p.radians(self.fov / 2)
            const result = self.brain.evaluate(sight.concat([1 - self.mass / self.maxMass]))
            const mag = self.baseSpeed + result[0]
            const direction = p.constrain(result[1], -halfFov, halfFov)

            self.neuralNetResults.push({magnitude: result[0], angle: result[1], frame: self.neuralNetResults.length})

            /*
             * Calculates the standard deviation for the rotation decisions of the shark in order to weed out those for which the outputs of the Neural Net are always the same whataver the input
             * Currently only doing self for the first 200 frames of the shark's life to reduce computing load
             */
            if (self.currentAge < 200) {
                self.angles.push(direction)
                const mean = self.angles.reduce((sum, value) => {return sum + value}, 0) / self.angles.length
                self.angleSD = Math.sqrt(self.angles.reduce((sum, value) => {return sum + (value - mean) * (value - mean)}, 0) / (self.angles.length - 1))
            }

            const computed = p5.Vector.fromAngle(self.velocity.heading(), mag)
            computed.rotate(direction)
            self.steer(computed, Infinity)

            if (debug) {
                computed.setMag(100)
                sketch.stroke(255, 0, 0)
                sketch.strokeWeight(2)
                sketch.line(self.position.x, self.position.y, self.position.x + computed.x, self.position.y + computed.y)
            }
        },

        eat: () => {
            const eaten = []
            const points = environment.qtree.query(new Circle(self.position.x, self.position.y, self.radius + 10), {types: ['foish']})

            for (const point of points) {
                const massGain = Math.min(point.data.mass, self.maxMass - self.mass)

                self.mass += massGain
                self.radius = 5 + self.mass
                self.score += massGain
                eaten.push(point.data.subject)
            }

            if (self.score > environment.frenzy.allTimeBest) {
                environment.frenzy.allTimeBest = self.score
            }

            return eaten
        },

        age: () => {
            self.currentAge++
        },

        hunger: () => {
            self.mass -= parseFloat(window.shoakHungerRate)

            // After 100 cycles of life, if the output of the Neural Net is constant and the shark keeps turning in a circle or just goes straight, accelerate it's death
            if (self.currentAge > 150 && self.angleSD < p.PI / 36) {
                self.mass -= 1
            }
            
            self.radius = getShoakRadius(self)
            self.shape = generateCircle(self)
        },

        reproduce: (id, parent) => {
            return Shoak(
                id,
                self.brain.crossover(parent.brain)
                //Math.abs(self.color - parent.color) > 60 ? Math.floor((self.color + parent.color + 360) / 2) % 360 : Math.floor((self.color + parent.color) / 2)
            )
        },

        cloneSelf: (id) => {
            return Shoak(id, self.brain.clone(), self.color)
        },

        mutate: (mutationRate) => {
            self.brain.mutate(mutationRate)
        },

        fitness: () => {
            // If the standard deviation of the rotation decisions of the Neural Net is below a certain value, reduce drastically the shark fitness to reduce its chances to be selected for reproduction
            const sd = p.map(self.angleSD, 0, p.PI / 36, 0, 1, true)

            return Math.pow(self.score * sd, 4)
        },

        show: (sketch) => {
            const opacity = p.map(null === environment.frenzy.aliveBest || 0 === environment.frenzy.aliveBest.score ? 1 : self.score / environment.frenzy.aliveBest.score, 0, 1, 50, 255, true)
            const shoakColor = p.color('hsba(' + self.color + ', 100%, 80%, ' + opacity + ')')

            sketch.stroke(shoakColor)
            sketch.fill(shoakColor)
            sketch.circle(Math.floor(self.position.x), Math.floor(self.position.y), self.radius * 2)
        }
    })

    return Object.assign(self, motileBehaviors(self), shoakBehaviors(self), canSee(self))
}

module.exports = Shoak
