import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss(), svelte()],
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
    format: "es", // engine worker imports split chunks; iife (default) can't
  },
  resolve: {
    alias: {
      // Sequence UI's ThemeProvider imports SvelteKit's $app/environment;
      // this app is plain Vite+Svelte, so shim it.
      "$app/environment": "/src/lib/sveltekit-shim.js",
      // The 'webgpu' npm package is Node/Dawn; browsers use navigator.gpu.
      webgpu: "/src/lib/webgpu-stub.ts",
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      // Node/Dawn inference server (server.ts) on 8787; SSE streams through fine.
      "/api": "http://127.0.0.1:8787",
    },
  },
});
