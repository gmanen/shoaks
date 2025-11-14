const Shoak = require('./motile/shoak')
const Foish = require('./motile/foish')
const Population = require('./evolution/population')
const {Point, Quadtree} = require('./quadtree/quadtree')
const {padding, topDownWidth, topDownHeight, povWidth, povHeight} = require('./config')
const environment = require ('./environment')

// Customizable parameters
window.displayPov = getParameter('displayPov', true)
window.displayStats = getParameter('displayStats', false)
window.frenzySize = getParameter('frenzySize', 30)
window.schoolSize = getParameter('schoolSize', 5)
window.shoakMutationRate = getParameter('shoakMutationRate', 10)
window.foishMutationRate = getParameter('foishMutationRate', 10)
window.shoakReproductionRate = getParameter('shoakReproductionRate', 0.1)
window.foishReproductionRate = getParameter('foishReproductionRate', 0.1)
window.shoakHungerRate = getParameter('shoakHungerRate', 0.08)
window.shoakNNComplexity = getParameter('shoakNNComplexity', 1)
window.foishNNComplexity = getParameter('foishNNComplexity', 1)
window.shoakNNSize = getParameter('shoakNNSize', 12)
window.foishNNSize = getParameter('foishNNSize', 12)
window.shoakSightRadius = getParameter('shoakSightRadius', 300)
window.foishSightRadius = getParameter('foishSightRadius', 200)
window.shoakFov = getParameter('shoakFov', 90)
window.foishFov = getParameter('foishFov', 300)
window.shoakResolution = getParameter('shoakResolution', 1)
window.foishResolution = getParameter('foishResolution', 1)
window.debug = getParameter('debug')

const p = new p5(sketch => {
    sketch.setup = function () {
        const canvas = sketch.createCanvas(topDownWidth, topDownHeight)
        canvas.parent('top-down-sketch')

        init()
    }

    sketch.draw = function () {
        sketch.background(175)

        if (environment.frenzy.isExtinct()) {
            for (const shoak of environment.frenzy.nextGeneration()) {
                environment.insert(shoak, 'shoak')
            }
        }

        if (environment.school.isExtinct()) {
            for (const foish of environment.school.nextGeneration()) {
                environment.insert(foish, 'foish')
            }
        }

        for (const shoak of environment.frenzy.reproduce()) {
            environment.insert(shoak, 'shoak')
        }

        for (const foish of environment.school.reproduce()) {
            environment.insert(foish, 'foish')
        }

        for (const shoak of environment.frenzy.population()) {
            for (const foish of shoak.eat()) {
                environment.school.remove(foish)
                environment.qtree.remove(foish.id)
            }

            shoak.bounce(padding, topDownWidth - padding, padding, topDownHeight - padding)
            shoak.think(sketch)
        }

        for (const foish of environment.school.population()) {
            foish.school()
            foish.bounce(padding, topDownWidth - padding, padding, topDownHeight - padding)
            foish.think(sketch)
        }

        // Done on the population instead of on every individual because it handles dying individuals and storing their data for later selection
        for (const shoak of environment.frenzy.hunger()) {
            environment.frenzy.remove(shoak)
            environment.qtree.remove(shoak.id)
        }

        environment.frenzy.age()
        environment.school.hunger()
        environment.school.age()

        for (const shoak of environment.frenzy.population()) {
            shoak.update()
            environment.qtree.move(shoak.id, shoak.position.x, shoak.position.y, shoak.shape, {
                mass: shoak.mass,
                position: shoak.position.copy(),
                velocity: shoak.velocity.copy()
            })
        }

        for (const foish of environment.school.population()) {
            foish.update()
            environment.qtree.move(foish.id, foish.position.x, foish.position.y, foish.shape, {
                mass: foish.mass,
                position: foish.position.copy(),
                velocity: foish.velocity.copy()
            })
        }

        sketch.strokeWeight(1)
        sketch.stroke(255)
        sketch.fill(255)
        sketch.text('Generation: ' + environment.frenzy.generation, 5, topDownHeight - 5)
        sketch.text('Alive: ' + environment.frenzy.population().length, 5, topDownHeight - 20)
        sketch.text('Current best: ' + (environment.frenzy.aliveBest ? environment.frenzy.aliveBest.score : 0).toFixed(2), 5, topDownHeight - 35)
        sketch.text('All time best: ' + environment.frenzy.allTimeBest.toFixed(2), 5, topDownHeight - 50)
        sketch.text('Frame rate: ' + Math.round(p.frameRate()), 5, topDownHeight - 65)

        // Drawing the top down scene
        for (const shoak of environment.frenzy.population()) {
            shoak.show(sketch)
        }

        for (const foish of environment.school.population()) {
            foish.show(sketch)
        }

        if (debug) {
            environment.qtree.show(sketch)
        }

        if (environment.frenzy.aliveBest) {
            p.stroke(255)
            p.strokeWeight(2)
            p.noFill()
            p.circle(environment.frenzy.aliveBest.position.x, environment.frenzy.aliveBest.position.y, environment.frenzy.aliveBest.radius * 2 + 1)
        }

        if (environment.school.aliveBest) {
            p.stroke(255)
            p.strokeWeight(2)
            p.noFill()
            p.circle(environment.school.aliveBest.position.x, environment.school.aliveBest.position.y, environment.school.aliveBest.radius * 2 + 1)
        }
    }
})
global.p = p

for (container of ['shoak-pov-sketch', 'foish-pov-sketch']) {
    new p5(sketch => {
        sketch.setup = function () {
            sketch.createCanvas(povWidth, povHeight)
        }

        sketch.draw = function () {
            sketch.background(175)

            let best = null

            if ('shoak' === sketch.canvas.parentElement.id.substr(0, 'shoak'.length)) {
                best = environment.frenzy.aliveBest
            } else if ('foish' === sketch.canvas.parentElement.id.substr(0, 'foish'.length)) {
                best = environment.school.aliveBest
            }

            if (best && window.displayPov) {
                const maxDist = best.sightRadius
                const w = povWidth / best.sight.length
                const maxDistSquared = maxDist * maxDist

                for (let i = 0; i < best.sight.length; i++) {
                    const {distance, color: pointColor} = best.sight[i]

                    if (distance >= 0) {
                        const distanceSquared = distance * distance
                        const light = Math.floor(sketch.map(distanceSquared, 0, maxDistSquared, 80, 20, true))
                        const h = sketch.map(distance, 0, maxDist, povHeight, 0, true)

                        sketch.noStroke()
                        sketch.fill(sketch.color('hsb(' + pointColor + ', 100%, ' + light + '%)'))
                        sketch.rectMode(sketch.CENTER)

                        sketch.rect(i * w + w / 2, povHeight / 2, w + 1, h)
                    }
                }
            }
        }
    }, container)
}

function init() {
    const frenzySize = debug ? 1 : parseInt(window.frenzySize)
    const schoolSize = debug ? 5 : frenzySize * parseInt(window.schoolSize)

    environment.qtree = new Quadtree(topDownWidth / 2, topDownHeight / 2, topDownWidth / 2, topDownHeight / 2, (frenzySize + schoolSize) / 10)

    environment.frenzy = new Population(
        frenzySize,
        parseFloat(window.shoakReproductionRate) / 100,
        parseFloat(window.shoakMutationRate) / 100,
        (id) => Shoak(id)
    )

    for (const shoak of environment.frenzy.populate()) {
        environment.qtree.insert(new Point(shoak.id, 'shoak', shoak.position.x, shoak.position.y, shoak.shape, {
            color: shoak.color,
            mass: shoak.mass,
            position: shoak.position.copy(),
            velocity: shoak.velocity.copy(),
            subject: shoak
        }))
    }

    environment.school = new Population(
        schoolSize,
        parseFloat(window.foishReproductionRate) / 100,
        parseFloat(window.foishMutationRate) / 100,
        (id) => Foish(id),
        schoolSize
    )

    for (const foish of environment.school.populate()) {
        environment.qtree.insert(new Point(foish.id, 'foish', foish.position.x, foish.position.y, foish.shape, {
            color: foish.color,
            mass: foish.mass,
            position: foish.position.copy(),
            velocity: foish.velocity.copy(),
            subject: foish
        }))
    }
}

function getParameter(name, defaultValue) {
    const value = window.sessionStorage.getItem(name)

    return null === value ? defaultValue || null : JSON.parse(value)
}

function setParameter(name, value) {
    window[name] = value

    return window.sessionStorage.setItem(name, JSON.stringify(value))
}

const chart = am4core.create("chart", am4charts.XYChart)
chart.xAxes.push(new am4charts.ValueAxis())
chart.yAxes.push(new am4charts.ValueAxis())
const magSeries = chart.series.push(new am4charts.LineSeries())
magSeries.name = "Magnitude"
magSeries.stroke = am4core.color("#0000FF")
magSeries.dataFields.valueY = "magnitude"
magSeries.dataFields.valueX = "frame"
const angleSeries = chart.series.push(new am4charts.LineSeries())
angleSeries.name = "Angle"
angleSeries.stroke = am4core.color("#FF0000")
angleSeries.dataFields.valueY = "angle"
angleSeries.dataFields.valueX = "frame"
chart.legend = new am4charts.Legend()

let currentCharted = 'aliveBest'

function refreshChart(toChart) {
    const chartTitle = document.getElementById('chart-title')

    if ('aliveBest' === toChart && environment.frenzy.aliveBest) {
        chart.data = environment.frenzy.aliveBest.neuralNetResults
        chartTitle.innerHTML = 'Best alive shoak (fitness: '+Math.pow(environment.frenzy.aliveBest.fitness(), 1/4).toFixed(2)+')'
    } else if ('aliveBest' !== toChart) {
        const corpses = environment.frenzy.graveyard.corpses.sort((a, b) => b.fitness() - a.fitness())

        if (corpses.length) {
            if (toChart >= corpses.length) {
                toChart = corpses.length - 1
            }

            chart.data = corpses[toChart].neuralNetResults
            chartTitle.innerHTML = 'Dead shoak '+(toChart+1)+' / '+corpses.length+' (fitness: '+Math.pow(corpses[toChart].fitness(), 1/4).toFixed(2)+')'
        }
    }
}

(function () {
    const sliders = [
        {variableName: 'frenzySize', sliderName: 'shoaks-population'},
        {variableName: 'shoakMutationRate', sliderName: 'shoaks-mutation-rate'},
        {variableName: 'shoakReproductionRate', sliderName: 'shoaks-reproduction-rate'},
        {variableName: 'shoakHungerRate', sliderName: 'shoaks-hunger-rate'},
        {variableName: 'shoakNNComplexity', sliderName: 'shoaks-nn-complexity'},
        {variableName: 'shoakNNSize', sliderName: 'shoaks-nn-size'},
        {variableName: 'shoakSightRadius', sliderName: 'shoaks-perception-radius'},
        {variableName: 'shoakFov', sliderName: 'shoaks-fov'},
        {variableName: 'shoakResolution', sliderName: 'shoaks-resolution'},
        {variableName: 'schoolSize', sliderName: 'foish-population'},
        {variableName: 'foishMutationRate', sliderName: 'foish-mutation-rate'},
        {variableName: 'foishReproductionRate', sliderName: 'foish-reproduction-rate'},
        {variableName: 'foishNNComplexity', sliderName: 'foish-nn-complexity'},
        {variableName: 'foishNNSize', sliderName: 'foish-nn-size'},
        {variableName: 'foishSightRadius', sliderName: 'foish-perception-radius'},
        {variableName: 'foishFov', sliderName: 'foish-fov'},
        {variableName: 'foishResolution', sliderName: 'foish-resolution'},
    ]

    document.getElementById('top-down-sketch').style.width = topDownWidth + 'px'
    const povContainer = document.getElementById('pov-container')
    povContainer.style.display = window.displayPov && !debug ? 'flex' : 'none'
    document.getElementById('display-pov').checked = window.displayPov
    document.getElementById('chart-container').style.display = window.displayStats && !debug ? 'block' : 'none'
    document.getElementById('display-stats').checked = window.displayStats

    for (const element of povContainer.children) {
        element.style.width = povWidth + 'px'
    }

    document.getElementById('debug').checked = debug
    document.getElementById('display-pov').disabled = debug
    document.getElementById('shoaks-population-slider').disabled = debug
    document.getElementById('chart-container').style.display = debug ? 'block' : 'none'

    document.getElementById('display-pov').addEventListener('change', (event) => {
        setParameter('displayPov', event.target.checked)
        document.getElementById('pov-container').style.display = event.target.checked ? 'flex' : 'none'
    })

    document.getElementById('display-stats').addEventListener('change', (event) => {
        setParameter('displayStats', event.target.checked)
        document.getElementById('chart-container').style.display = event.target.checked ? 'block' : 'none'
    })

    document.getElementById('debug').addEventListener('change', (event) => {
        setParameter('debug', event.target.checked)
        document.getElementById('shoaks-population-slider').disabled = event.target.checked
        debug = event.target.checked

        document.getElementById('display-pov').disabled = debug
        document.getElementById('display-stats').disabled = debug

        document.getElementById('pov-container').style.display = debug || !window.displayPov ? 'none' : 'flex'
        document.getElementById('chart-container').style.display = debug ? 'block' : 'none'

        if (debug) {
            refreshChart(currentCharted)
        }

        init()
    })

    for (const slider of sliders) {
        const currentValue = window[slider.variableName]

        for (const element of document.getElementsByClassName(slider.sliderName + '-current')) {
            element.innerHTML = currentValue
        }

        document.getElementById(slider.sliderName + '-slider').value = currentValue

        document.getElementById(slider.sliderName + '-slider').addEventListener('change', (event) => {
            setParameter(slider.variableName, event.target.value)

            for (const element of document.getElementsByClassName(slider.sliderName + '-current')) {
                element.innerHTML = event.target.value
            }

            init()
        })
    }

    document.getElementById('chart-prev').addEventListener('click', event => {
        event.preventDefault()

        if ('aliveBest' === currentCharted) {
            currentCharted = environment.frenzy.graveyard.corpses.length - 1
        } else {
            currentCharted--

            if (currentCharted < 0) {
                currentCharted = 'aliveBest'
            }
        }

        refreshChart(currentCharted)
    })

    document.getElementById('chart-next').addEventListener('click', event => {
        event.preventDefault()

        if ('aliveBest' === currentCharted) {
            currentCharted = 0
        } else {
            currentCharted++

            if (currentCharted >= environment.frenzy.graveyard.corpses.length) {
                currentCharted = 'aliveBest'
            }
        }

        refreshChart(currentCharted)
    })

    document.getElementById('chart-refresh').addEventListener('click', event => {
        event.preventDefault()

        refreshChart(currentCharted)
    })
})()
