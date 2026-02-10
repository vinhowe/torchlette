import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  optimizeDeps: {
    exclude: ['torchlette']
  },
  resolve: {
    alias: {
      // The 'webgpu' npm package is for Node.js (Dawn). In browsers, we use native WebGPU.
      // Replace with an empty module to avoid bundling Node.js-specific code.
      'webgpu': '/src/lib/webgpu-stub.ts'
    }
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      external: []
    }
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  }
});
