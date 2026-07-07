import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit(), tailwindcss()],
  define: {
    "process.env": JSON.stringify({}),
  },
  optimizeDeps: {
    exclude: ["torchlette"],
  },
  resolve: {
    alias: {
      // The 'webgpu' npm package is for Node.js (Dawn). In browsers, we use native WebGPU.
      // Replace with an empty module to avoid bundling Node.js-specific code.
      webgpu: "/src/lib/webgpu-stub.ts",
    },
  },
  build: {
    target: "esnext",
  },
  server: {
    headers: {
      // Cross-origin isolation: required for SharedArrayBuffer / high-precision timers
      // the torchlette WebGPU backend may use. HF endpoints send CORS, so cross-origin
      // fetches (huggingface.co, datasets-server) still work under require-corp.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
