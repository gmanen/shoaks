const {motileBehaviors} = require('./motile')
const {canSee} = require('./sighted')
const Brain = require('./brain')
const {topDownWidth, topDownHeight} = require('../config')
const {Circle} = require('../quadtree/quadtree')
const environment = require('../environment')

const generatePoly = self => {
    const poly = []
    const head = p5.Vector.fromAngle(self.velocity.heading(), self.radius)

    poly.push({x: self.position.x + head.x, y: self.position.y + head.y})
    head.rotate(self.angle)
    poly.push({x: self.position.x + head.x, y: self.position.y + head.y})
    head.rotate(-2 * self.angle)
    poly.push({x: self.position.x + head.x, y: self.position.y + head.y})

    return poly
}

const Foish = (id, brain, foishColor) => {
    const baseSpeed = 4
    const baseMass = p.random(2, 4)
    const velocity = p5.Vector.random2D()
    velocity.setMag(baseSpeed)

    const fov = parseInt(window.foishFov)
    const resolution = parseFloat(window.foishResolution)

    if (!brain) {
        const layers = []

        for (let i = 0; i < parseInt(window.foishNNComplexity); i++) {
            layers.push(parseInt(window.foishNNSize))
        }

        brain = new Brain(fov * resolution, layers, 2, 'relu')
        brain.randomize()
    }

    const self = {
        id: 'foish-'+id,
        baseSpeed,
        minSpeed: 1,
        maxSpeed: 8,
        maxForce: 0.2,
        mass: baseMass,
        position: p.createVector(p.random(topDownWidth), p.random(topDownHeight)),
        velocity: velocity,
        acceleration: p.createVector(),
        radius: 2 * baseMass,
        alignPerceptionRadius: 60,
        cohesionPerceptionRadius: 75,
        separationPerceptionRadius: 30,
        sightRadius: parseInt(window.foishSightRadius),
        alignWeight: 1,
        cohesionWeight: 1,
        separationWeight: 1.5,
        flockingFov: 8 * p.PI / 10,
        fov,
        resolution,
        angle: 5 * p.PI / 6,
        sight: [],
        brain,
        currentAge: 0,
        score: 0,
        color: foishColor || 240 + Math.floor(p.random(-30, 31)),
        generateShape: generatePoly
    }

    self.shape = generatePoly(self)

    const foishBehaviors = self => ({
        school: () => {
            const align = self.align(environment.qtree)
            const cohesion = self.cohesion(environment.qtree)
            const separation = self.separation(environment.qtree)
            const isAlign = align.x !== 0 && align.y !== 0
            const isCohesion = cohesion.x !== 0 && cohesion.y !== 0
            const isSeparation = separation.x !== 0 && separation.y !== 0

            if (isAlign) {
                self.steer(align)
            }

            if (isCohesion) {
                self.steer(cohesion)
            }

            if (isSeparation) {
                self.steer(separation)
            }

            if (!(isAlign || isCohesion || isSeparation)) {
                self.steer(p5.Vector.fromAngle(self.velocity.heading(), self.baseSpeed))
            }
        },

        align: () => {
            const alignment = p.createVector()
            let alignmentTotal = 0

            for (const point of environment.qtree.query(new Circle(self.position.x, self.position.y, self.alignPerceptionRadius), {types: ['foish']})) {
                if (point.id === self.id) {
                    continue
                }

                if (self.velocity.angleBetween(p5.Vector.sub(self.position, point.data.position)) > self.flockingFov) {
                    continue
                }

                alignment.add(point.data.velocity)
                alignmentTotal++
            }

            if (alignmentTotal > 0) {
                alignment.div(alignmentTotal)
                alignment.setMag(self.baseSpeed)
                alignment.mult(self.alignWeight)
            }

            return alignment
        },

        cohesion: () => {
            const cohesion = p.createVector()
            let cohesionTotal = 0

            for (let point of environment.qtree.query(new Circle(self.position.x, self.position.y, self.cohesionPerceptionRadius), {types: ['foish']})) {
                if (point.id === self.id) {
                    continue
                }

                const otherPosition = point.data.position

                if (self.velocity.angleBetween(p5.Vector.sub(self.position, otherPosition)) > self.flockingFov) {
                    continue
                }

                cohesion.add(otherPosition)
                cohesionTotal++
            }

            if (cohesionTotal > 0) {
                cohesion.div(cohesionTotal)
                cohesion.sub(self.position)
                cohesion.setMag(self.baseSpeed)
                cohesion.mult(self.cohesionWeight)
            }

            return cohesion
        },

        separation: () => {
            const separation = p.createVector()
            let separationTotal = 0

            for (let point of environment.qtree.query(new Circle(self.position.x, self.position.y, self.separationPerceptionRadius), {types: ['foish']})) {
                if (point.id === self.id) {
                    continue
                }

                const otherPosition = point.data.position
                const diff = p5.Vector.sub(self.position, otherPosition)

                if (self.velocity.angleBetween(diff) > self.flockingFov) {
                    continue
                }

                const d = p.dist(self.position.x, self.position.y, otherPosition.x, otherPosition.y)
                const dSquared = d * d

                if (0 !== dSquared) {
                    diff.div(dSquared)
                    separation.add(diff)

                    separationTotal++
                }
            }

            if (separationTotal > 0) {
                separation.div(separationTotal)
                separation.setMag(self.baseSpeed)
                separation.mult(self.separationWeight)
            }

            return separation
        },

        think: (sketch) => {
            const sight = self.see(environment.qtree, ['shoak'], self.sight, sketch)

            if (debug) {
                const velocity = p5.Vector.fromAngle(self.velocity.heading(), 100)
                sketch.stroke(0, 0, 255)
                sketch.strokeWeight(2)
                sketch.line(self.position.x, self.position.y, self.position.x + velocity.x, self.position.y + velocity.y)
            }

            if (sight.reduce((total, currentValue) => total + currentValue, 0)) {
                const halfFov = p.radians(self.fov / 2)
                const result = self.brain.evaluate(sight)
                const computed = p5.Vector.fromAngle(self.velocity.heading(), p.map(p.atan(result[0] / p.PI), -0.5, 0.5, self.minSpeed, self.maxSpeed))

                computed.rotate(p.map(p.atan(result[1]) / p.PI, -0.5, 0.5, -halfFov, halfFov))
                self.steer(computed, 2 * self.maxForce)

                if (debug) {
                    computed.setMag(50)
                    sketch.stroke(255, 0, 0)
                    sketch.strokeWeight(2)
                    sketch.line(self.position.x, self.position.y, self.position.x + computed.x, self.position.y + computed.y)
                }
            }
        },

        age: () => {
            self.currentAge++
            self.score++
        },

        hunger: () => {

        },

        reproduce: (id, parent) => {
            return Foish(id, self.brain.crossover(parent.brain), Math.floor((self.color + parent.color) / 2))
        },

        cloneSelf: (id) =>  {
            return Foish(id, self.brain.clone(), self.color)
        },

        mutate: (mutationRate) => {
            self.brain.mutate(mutationRate)
        },

        fitness: () => {
            return Math.pow(self.score, 4)
        },

        show: (sketch) => {
            const foishColor = p.color('hsba(' + self.color + ', 100%, 80%, 1)')

            sketch.strokeWeight(1)
            sketch.stroke(foishColor)
            sketch.fill(foishColor)
            sketch.beginShape(p.TRIANGLES)

            for (const point of self.shape) {
                sketch.vertex(point.x, point.y)
            }

            sketch.endShape(p.CLOSE)
        }
    })

    return Object.assign(self, motileBehaviors(self), foishBehaviors(self), canSee(self))
}

module.exports = Foish
