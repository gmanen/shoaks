const Graveyard = require('./graveyard')

class Population {
    constructor(size, reproductionRate, mutationRate, generateIndividual) {
        this.size = size
        this.reproductionRate = reproductionRate
        this.mutationRate = mutationRate
        this.generateIndividual = generateIndividual

        this.individuals = []
        this.graveyard = new Graveyard()
        this.allTimeBest = 0
        this.aliveBest = null
        this.generation = 1
        this.nextId = 1
    }

    populate() {
        const created = []

        for (let i = this.individuals.length; i < this.size; i++) {
            const newIndividual = this.generateIndividual(this.nextId)
            this.nextId++

            this.individuals.push(newIndividual)
            created.push(newIndividual)
        }

        return created
    }

    population() {
        return this.individuals
    }

    isExtinct() {
        return 0 === this.individuals.length
    }

    nextGeneration() {
        if (!this.isExtinct()) {
            return []
        }

        let created = []
        const corpses = this.graveyard.corpses

        for (let i = 0; i < this.size; i++) {
            if ((corpses.length < 2) || p.random() <= 0.5) {
                created = created.concat(this.reproduceAsexually(this.select(1, corpses)))
            } else {
                created = created.concat(this.reproduceSexually([this.select(2, corpses, true)]))
            }
        }

        this.graveyard.clear()
        this.generation++

        return created
    }

    age() {
        this.aliveBest = null

        for (const individual of this.individuals) {
            individual.age()

            if (null === this.aliveBest || individual.score > this.aliveBest.score) {
                this.aliveBest = individual
            }
        }
    }

    hunger() {
        const starved = []

        for (let i = this.individuals.length - 1; i >=0; i--) {
            this.individuals[i].hunger()

            if (this.individuals[i].mass <= 0) {
                starved.push(this.individuals[i])
            }
        }

        return starved
    }

    remove(individual) {
        const index = this.individuals.indexOf(individual)

        if (index >= 0) {
            this.graveyard.addCorpse(this.individuals.splice(index, 1)[0])
        }
    }

    reproduce() {
        if (this.isExtinct()) {
            return []
        }

        let created = []

        if (this.individuals.length < this.size) {
            for (let i = 0; i < this.size - this.individuals.length; i++) {
                const roll = p.random()

                if (roll < this.reproductionRate) {
                    if ((this.individuals.length < 2) || roll <= this.reproductionRate / 2) {
                        created = created.concat(this.reproduceAsexually(this.select(1, this.individuals)))
                    } else {
                        created = created.concat(this.reproduceSexually([this.select(2, this.individuals, true)]))
                    }
                }
            }
        }

        return created
    }

    reproduceAsexually(individuals) {
        const created = []

        for (const individual of individuals) {
            const child = individual.cloneSelf(this.nextId++)
            child.mutate(this.mutationRate)

            this.individuals.push(child)
            created.push(child)
        }

        return created
    }

    reproduceSexually(couples) {
        const created = []

        for (const individuals of couples) {
            const child = individuals[0].reproduce(this.nextId++, individuals[1])

            child.mutate(this.mutationRate)

            this.individuals.push(child)
            created.push(child)
        }

        return created
    }

    select(number, group, different) {
        if (number > group.length) {
            throw 'Cannot select '+number+' individuals from '+group.length
        }

        const population = [...group]
        const selectedList = []
        let sumFitness = 0
        let minFitness = Infinity
        let maxFitness = 0

        for (let individual of population) {
            const fitness  = individual.fitness()

            sumFitness += fitness

            if (fitness < minFitness) minFitness = fitness
            if (fitness > maxFitness) maxFitness = fitness
        }

        if (debug) {
            console.log('Selecting '+number+' individuals out of '+ population.length + ' (max fitness = '+Math.pow(maxFitness, 1/4).toFixed(2)+', min fitness = '+Math.pow(minFitness, 1/4).toFixed(2)+')')
        }

        for (let i = 0; i < number; i++) {
            const random = p.random()
            let offset = population[0].fitness()
            let selected = population[0]

            for (let j = 1; j < population.length && population[0] === selected; j++) {
                offset += population[j].fitness() / sumFitness

                if (random <= offset) {
                    selected = population[j]
                }
            }

            if (different) {
                population.splice(population.indexOf(selected), 1)
                sumFitness -= selected.fitness()
            }

            if (debug) {
                console.log('Selected fitness '+selected.score.toFixed(2))
            }

            selectedList.push(selected)
        }

        return selectedList
    }
}

module.exports = Population
