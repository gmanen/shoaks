import {Circle, Line} from '../quadtree/quadtree.js'

export const canSee = (self) => ({
    // storedSight should be an array. It will be reset and then filled with the distance values of each closest point colliding with one of the rays to display the POV scene
    see: (qtree, types = [], storedSight, sketch) => {
        if (Array.isArray(storedSight)) {
            storedSight.length = 0
        }

        const sight = []

        const angleStepDegrees = 1 / self.resolution
        const angleStep = p.radians(angleStepDegrees)

        const rayVectors = []
        const rayOffsets = []

        for (let angle = -self.fov / 2; angle < self.fov / 2; angle += angleStepDegrees) {
            const offset = p.radians(angle)

            rayOffsets.push(offset)
            rayVectors.push(p5.Vector.fromAngle(offset, 1))
        }

        const heading = self.velocity.heading()
        const angleDelta = angleStep / 2

        const points = qtree.query(
            new Circle(self.position.x, self.position.y, self.sightRadius),
            {exclude: [self.id], types}
        )

        const normalizeAngle = (angle) => {
            const doublePi = Math.PI * 2
            let normalized = angle

            while (normalized > Math.PI) {
                normalized -= doublePi
            }

            while (normalized <= -Math.PI) {
                normalized += doublePi
            }

            return normalized
        }

        const candidates = points
            .map((point) => {
                const dx = point.x - self.position.x
                const dy = point.y - self.position.y
                const relativeAngle = normalizeAngle(p.atan2(dy, dx) - heading)
                const radius = point.shape && point.shape.r ? point.shape.r : 0
                const centerDistance = p.dist(self.position.x, self.position.y, point.x, point.y)

                return {point, relativeAngle, radius, centerDistance}
            })
            .sort((a, b) => a.relativeAngle - b.relativeAngle)

        const findStartIndex = (targetAngle) => {
            let low = 0
            let high = candidates.length

            while (low < high) {
                const mid = (low + high) >> 1

                if (candidates[mid].relativeAngle < targetAngle) {
                    low = mid + 1
                } else {
                    high = mid
                }
            }

            return low
        }

        for (let i = 0; i < rayVectors.length; i++) {
            const rayVector = rayVectors[i].copy().rotate(heading)
            const rayDirection = rayVector.copy().setMag(self.sightRadius)
            const rayLine = new Line(
                self.position,
                p.createVector(self.position.x + rayDirection.x, self.position.y + rayDirection.y)
            )
            let closest = self.sightRadius
            let closestPoint = null
            let intersectingColor = null
            let hasHit = false

            const rayAngle = rayOffsets[i]
            const minAngle = rayAngle - angleDelta
            const maxAngle = rayAngle + angleDelta
            const startIndex = findStartIndex(minAngle)

            for (let idx = startIndex; idx < candidates.length && candidates[idx].relativeAngle <= maxAngle; idx++) {
                const {point, radius, centerDistance} = candidates[idx]

                if (centerDistance - radius > self.sightRadius || centerDistance - radius >= closest) {
                    continue
                }

                for (const intersectingPoint of rayLine.intersectsShape(point.shape)) {
                    const d = p.dist(self.position.x, self.position.y, intersectingPoint.x, intersectingPoint.y)

                    if (d < closest) {
                        closest = d
                        closestPoint = intersectingPoint
                        intersectingColor = point.data.color
                        hasHit = true
                    }
                }
            }

            const distance = hasHit && closest <= self.sightRadius ? closest : self.sightRadius

            if (debug) {
                const drawRay = p5.Vector.fromAngle(rayVector.heading(), distance)

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
                const projectionAngle = rayVector.heading() - self.velocity.heading()

                if (projectionAngle <= p.PI / 4 && projectionAngle >= -p.PI / 4) {
                    storedSight.push({
                        'distance': hasHit ? (distance * p.cos(rayVector.heading() - self.velocity.heading())) : -1,
                        'color': intersectingColor
                    })
                }
            }
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
