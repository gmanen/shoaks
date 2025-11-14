const {Point} = require('./quadtree/quadtree')

class Environment {
    constructor() {
        this.frenzy = null
        this.school = null
        this.qtree = null
    }

    insert(subject, type) {
        this.qtree.insert(new Point(subject.id, type, subject.position.x, subject.position.y, subject.shape, {
            color: subject.color,
            mass: subject.mass,
            position: subject.position.copy(),
            velocity: subject.velocity.copy(),
            subject: subject
        }))
    }
}
const environment = new Environment()

module.exports = environment
