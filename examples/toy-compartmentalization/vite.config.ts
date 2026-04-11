import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import path from "node:path";

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  define: {
    "process.env": JSON.stringify({}),
  },
  optimizeDeps: {
    exclude: ["torchlette"],
  },
  resolve: {
    alias: {
      // Resolve torchlette from source (Vite handles TS natively in dev mode)
      torchlette: path.resolve(__dirname, "../../src/browser.ts"),
      webgpu: "/src/lib/webgpu-stub.ts",
    },
  },
  build: {
    target: "esnext",
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
