import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    rollupOptions: {
      input: 'src/sketch.js',
    },
  },
  optimizeDeps: {
    entries: ['src/sketch.js'],
  },
})
