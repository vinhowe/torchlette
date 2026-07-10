import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit(), tailwindcss()],
  define: {
    "process.env": JSON.stringify({}),
  },
  optimizeDeps: {
    exclude: ["torchlette", "gemma2-browser", "gemma-scope-sae"],
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
    // 5173/5174/5175 are taken by the other torchlette demos.
    port: 5176,
    strictPort: true,
    watch: {
      // The workspace packages are resolved through node_modules symlinks,
      // which Vite ignores for HMR by default — so edits to gemma2-browser /
      // torchlette source were never re-served (the dev server pinned the
      // first transform of each @fs module). Un-ignore them so source edits
      // to the linked packages hot-reload instead of requiring a restart.
      ignored: [
        "!**/packages/gemma2-browser/**",
        "!**/packages/gemma-scope-sae/**",
      ],
    },
  },
});
