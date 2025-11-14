const {Circle, Line} = require('../quadtree/quadtree')

const canSee = self => ({
    // storedSight should be an array. It will be reset and then filled with the distance values of each closest point colliding with one of the rays to display the POV scene
    see: (qtree, types, storedSight, sketch) => {
        types = types || []

        if (Array.isArray(storedSight)) {
            storedSight.length = 0
        }

        const sight = []
        const angleVector = p5.Vector.fromAngle(self.velocity.heading(), 1)
        angleVector.rotate(p.radians(-self.fov / 2)) // Starting angle for the rays

        const points = qtree.query(
            new Circle(self.position.x, self.position.y, self.sightRadius),
            {exclude: [self.id], types}
        )

        for (let i = 0; i < self.fov; i += (1 / self.resolution)) {
            const rayLine = new Line(self.position, p.createVector(self.position.x + angleVector.x, self.position.y + angleVector.y))
            let closest = Infinity
            let closestPoint = null
            let intersectingColor = null

            for (const point of points) {
                for (const intersectingPoint of rayLine.intersectsShape(point.shape)) {
                    const d = p.dist(self.position.x, self.position.y, intersectingPoint.x, intersectingPoint.y)

                    if (d < closest) {
                        closest = d
                        closestPoint = intersectingPoint
                        intersectingColor = point.data.color
                    }
                }
            }

            const distance = Infinity === closest || closest > self.sightRadius ? self.sightRadius : closest

            if (debug) {
                const drawRay = p5.Vector.fromAngle(angleVector.heading(), closest === Infinity ? self.sightRadius : closest)

                sketch.stroke(255, 255, 255, 20)
                sketch.strokeWeight(5)
                sketch.line(self.position.x, self.position.y, self.position.x + drawRay.x, self.position.y + drawRay.y)

                if (closestPoint) {
                    sketch.strokeWeight(1)
                    sketch.stroke(255, 0, 0)
                    sketch.fill(255, 0, 0)
                    sketch.circle(closestPoint.x, closestPoint.y, 2)
                }
            }

            /*
             * Input for the shark's Neural Net, for each ray the distance to the closest fish is a value from 0 to 1
             * The closest fishes will have a value closer to 1, furthest a value closer to 0
             */
            sight.push(p.map(distance, 0, self.sightRadius, 1, 0, true))

            if (Array.isArray(storedSight)) { // the results can be displayed in a POV Scene
                // -1 means no fish intersects that ray so nothing should be displayed in the POV scene
                const projectionAngle = angleVector.heading() - self.velocity.heading()

                if (projectionAngle <= p.PI / 4 && projectionAngle >= -p.PI / 4) {
                    storedSight.push({
                        'distance': Infinity === closest ? -1 : (distance * p.cos(angleVector.heading() - self.velocity.heading())),
                        'color': intersectingColor
                    })
                }
            }

            angleVector.rotate(p.radians(1 / self.resolution))
        }

        if (debug) {
            sketch.noFill()
            sketch.strokeWeight(1)
            sketch.stroke(255)
            sketch.circle(self.position.x, self.position.y, self.sightRadius * 2)
        }

        return sight
    }
})

module.exports.canSee = canSee
