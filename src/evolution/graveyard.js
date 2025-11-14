export default class Graveyard {
    constructor() {
        this.corpses = []
    }

    addCorpse(corpse) {
        this.corpses.push(corpse)
    }

    clear() {
        this.corpses = []
    }
}
