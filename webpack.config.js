const path = require('path');

module.exports = {
    entry: {
        'sketch': './src/sketch.js',
    },
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: '[name].js'
    },
    module: {
        noParse: [
            /benchmark/,
        ]
    },
}
