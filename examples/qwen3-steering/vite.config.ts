import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit(), tailwindcss()],
  define: {
    "process.env": JSON.stringify({}),
  },
  optimizeDeps: {
    exclude: ["torchlette", "qwen3-browser"],
  },
  build: {
    target: "esnext",
  },
  worker: {
    // The engine worker imports split chunks; iife (the default) can't.
    format: "es",
  },
  resolve: {
    alias: {
      // The 'webgpu' npm package is Node/Dawn; browsers use navigator.gpu.
      webgpu: "/src/lib/webgpu-stub.ts",
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5175,
    strictPort: true,
  },
});
