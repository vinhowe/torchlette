import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit()],
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
    rollupOptions: {
      external: [],
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    proxy: {
      // PeerJS signaling server on host (sivri) via Docker gateway
      "/peer-signal": {
        target: "http://172.17.0.1:9000",
        ws: true,
        rewrite: (path: string) => path.replace(/^\/peer-signal/, ""),
      },
      // WebSocket relay for Docker↔host gradient bridge
      "/ws-relay": {
        target: "ws://172.17.0.1:9876",
        ws: true,
        rewrite: (path: string) => path.replace(/^\/ws-relay/, ""),
      },
    },
  },
});
