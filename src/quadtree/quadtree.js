class Point {
    constructor(id, type, x, y, shape, data) {
        this.id = id
        this.type = type
        this.x = x
        this.y = y
        this.shape = shape
        this.data = data
        this.owner = null
    }
}

class Line {
    constructor(point1, point2) {
        this.point1 = point1
        this.point2 = point2
    }

    intersects(line) {
        const x1 = line.point1.x
        const y1 = line.point1.y
        const x2 = line.point2.x
        const y2 = line.point2.y

        const x3 = this.point1.x
        const y3 = this.point1.y
        const x4 = this.point2.x
        const y4 = this.point2.y

        const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if (0 === den) {
            return false
        }

        const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if (!(t > 0 && t < 1 && u > 0)) {
            return false;
        }

        return {x: x1 + t * (x2 - x1), y: y1 + t * (y2 - y1)};
    }

    intersectsShape(shape) {
        if (shape instanceof Circle) {
            return this.intersectsCircle(shape)
        }

        return this.intersectsPoly(shape)
    }

    intersectsPoly(poly) {
        const intersects = []

        for (let i = 0; i < poly.length; i++) {
            const intersect = this.intersects(new Line(poly[i], poly[(i + 1) % poly.length]))

            if (false !== intersect) {
                intersects.push(intersect)
            }
        }

        return intersects.length ? intersects : []
    }

    intersectsCircle(circle) {
        const r = circle.r
        const h = circle.x
        const k = circle.y

        const m = (this.point2.y - this.point1.y) / (this.point2.x - this.point1.x)
        const n = this.point1.y - m * this.point1.x

        const a = 1 + p.sq(m)
        const b = -h * 2 + (m * (n - k)) * 2
        const c = p.sq(h) + p.sq(n - k) - p.sq(r)
        const d = p.sq(b) - 4 * a * c

        if (d >= 0) {
            const intersections = [
                (-b + p.sqrt(p.sq(b) - 4 * a * c)) / (2 * a),
                (-b - p.sqrt(p.sq(b) - 4 * a * c)) / (2 * a)
            ]

            if (0 === d && intersections[0] > this.point1.x === this.point2.x > this.point1.x) {
                return [{x: intersections[0], y: m * intersections[0] + n}]
            }

            if (0 === d) {
                return []
            }

            const result = []

            for (const intersection of intersections) {
                if (intersection > this.point1.x === this.point2.x > this.point1.x) {
                    result.push({x: intersection, y: m * intersection + n})
                }
            }

            return result
        }

        return [];
    }
}

class Rectangle {
    constructor(x, y, w, h) {
        this.x = x
        this.y = y
        this.w = w
        this.h = h
    }

    intersects(circle) {
        let dx = Math.abs(this.x - circle.x)
        let dy = Math.abs(this.y - circle.y)

        const r = circle.r
        const w = this.w
        const h = this.h

        if (dx > (r + w) || dy > (r + h)) {
            return false
        }

        if (dx <= w || dy <= h) {
            return true
        }

        dx = dx - w
        dy = dy - h

        return dx * dx + dy * dy <= r * r
    }

    intersectsLine(line) {
        const nw = {x: this.x - this.w, y: this.y - this.h}
        const ne = {x: this.x + this.w, y: this.y - this.h}
        const sw = {x: this.x - this.w, y: this.y + this.h}
        const se = {x: this.x + this.w, y: this.y + this.h}

        return line.intersects(new Line(nw, ne)) || line.intersects(new Line(ne, se)) || line.intersects(new Line(se, sw)) || line.intersects(new Line(sw, nw))
    }

    contains(point) {
        const x = this.x
        const y = this.y
        const w = this.w
        const h = this.h

        return !(point.x > x + w || point.x < x - w || point.y < y - h || point.y > y + h)
    }
}

class Circle {
    constructor(x, y, r) {
        this.x = x
        this.y = y
        this.r = r
    }

    contains(point) {
        const dx = Math.abs(this.x - point.x)
        const dy = Math.abs(this.y - point.y)

        return dx * dx + dy * dy <= this.r * this.r
    }
}

class Quadtree {
    constructor(x, y, w, h, capacity, parent, root) {
        this.boundary = new Rectangle(x, y, w, h)
        this.capacity = capacity
        this.points = {}
        this.localPoints = []
        this.regions = []
        this.parent = parent || null
        this.root = root || this
    }

    insert(point) {
        if (!this.boundary.contains(point)) {
            return false
        }

        this.points[point.id] = point

        this.insertRecusive(point.id)
    }

    insertRecusive(pointId) {
        if (!this.boundary.contains(this.root.points[pointId])) {
            return false
        }

        if (!this.isSubdivided() && this.localPoints.length < this.capacity) {
            this.localPoints.push(pointId)
            this.root.points[pointId].owner = this
            return true
        }

        this.subdivide()

        for (const insertPoint of [...this.localPoints, pointId]) {
            for (const region of this.regions) {
                if (region.insertRecusive(insertPoint)) {
                    break
                }
            }
        }

        this.localPoints = []

        return true
    }

    subdivide() {
        if (this.isSubdivided()) {
            return
        }

        const x = this.boundary.x
        const y = this.boundary.y
        const w = this.boundary.w / 2
        const h = this.boundary.h / 2

        this.regions.push(new Quadtree(x - w, y - h, w, h, this.capacity, this, this.root))
        this.regions.push(new Quadtree(x + w, y - h, w, h, this.capacity, this, this.root))
        this.regions.push(new Quadtree(x - w, y + h, w, h, this.capacity, this, this.root))
        this.regions.push(new Quadtree(x + w, y + h, w, h, this.capacity, this, this.root))
    }

    query(circle, options) {
        options = Object.assign({exclude: [], types: []}, options)

        return Object.values(this.queryRecursive(circle, options))
    }

    queryRecursive(circle, options) {
        if (!(this.boundary.intersects(circle))) {
            return {}
        }

        let results = {}

        if (this.isSubdivided()) {
            for (const region of this.regions) {
                results = Object.assign(results, region.queryRecursive(circle, options))
            }

            return results
        }

        for (const pointId of this.localPoints) {
            if (options.exclude.indexOf(pointId) >= 0) {
                continue
            }

            if (options.types.length > 0 && options.types.indexOf(this.root.points[pointId].type) < 0) {
                continue
            }

            const point = this.root.points[pointId]

            if (circle.contains(point)) {
                results[pointId] = point
            }
        }

        return results
    }

    move(pointId, newX, newY, newShape, newData) {
        const point = this.root.points[pointId]

        point.x = newX
        point.y = newY
        point.shape = newShape

        if (newData) {
            point.data = Object.assign(point.data, newData)
        }

        if (point.owner.boundary.contains(point)) {
            return
        }

        const formerOwner = point.owner

        if (null !== formerOwner.parent) {
            formerOwner.parent.relocate(point)
            formerOwner.localPoints.splice(formerOwner.localPoints.indexOf(pointId), 1)
            formerOwner.parent.cleanup()
        }
    }

    relocate(point) {
        if (!this.boundary.contains(point)) {
            if (null !== this.parent) {
                this.parent.relocate(point)
            }

            return
        }

        this.insertRecusive(point.id)
    }

    cleanup() {
        let points = this.getLocalPointsRecursive()

        if (points.length < this.capacity) {
            this.localPoints = points

            for (const point of points) {
                this.root.points[point].owner = this
            }

            this.regions = []

            if (null !== this.parent) {
                this.parent.cleanup()
            }
        }
    }

    remove(pointId) {
        const owner = this.points[pointId].owner

        owner.localPoints.splice(owner.localPoints.indexOf(pointId), 1)

        if (null !== owner.parent) {
            owner.parent.cleanup()
        }

        delete this.points[pointId]
    }

    queryLine(line) {
        if (!this.boundary.intersectsLine(line)) {
            return []
        }

        let results = []

        if (this.isSubdivided()) {
            for (const region of this.regions) {
                results = results.concat(region.queryLine(line))
            }

            return results
        }

        for (const pointId of this.localPoints) {
            const point = this.root.points[pointId]
            const intersectingPoints = line.intersectsShape(point.shape)

            if (false !== intersectingPoints) {
                results = results.concat(intersectingPoints)
            }
        }

        return results
    }

    count() {
        let count = 0

        if (this.isSubdivided()) {
            for (const region of this.regions) {
                count += region.count()
            }

            return count
        }

        return this.localPoints.length
    }

    getLocalPointsRecursive() {
        let points = []

        if (this.isSubdivided()) {
            for (const region of this.regions) {
                points = points.concat(region.getLocalPointsRecursive())
            }
        }

        return points.concat(this.localPoints)
    }

    isSubdivided() {
        return this.regions.length > 0
    }

    show(sketch) {
        sketch.stroke(255)
        sketch.strokeWeight(1)
        sketch.noFill()
        sketch.rectMode(sketch.CENTER)
        sketch.rect(this.boundary.x, this.boundary.y, this.boundary.w * 2, this.boundary.h * 2)

        if (this.isSubdivided()) {
            this.regions.map(region => {
                region.show(sketch)
            })
        } else {
            sketch.text(this.count(), this.boundary.x, this.boundary.y)
        }
    }
}

module.exports.Point = Point
module.exports.Line = Line
module.exports.Rectangle = Rectangle
module.exports.Circle = Circle
module.exports.Quadtree = Quadtree
