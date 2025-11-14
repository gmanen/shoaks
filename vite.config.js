import { defineConfig } from 'vite'
import commonjs from '@originjs/vite-plugin-commonjs'

export default defineConfig({
  plugins: [commonjs()],
  build: {
    rollupOptions: {
      input: 'src/sketch.js',
    },
  },
  optimizeDeps: {
    entries: ['src/sketch.js'],
  },
})
