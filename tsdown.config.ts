import { defineConfig } from "tsdown";

export default defineConfig([
  {
    // Node entry + the `nn` subpath ("torchlette/nn"). Exposing nn as its own
    // named entry gives consumers a properly-typed module of named exports —
    // the main entry's `export * as nn` namespace re-export does not survive
    // into the bundled .d.ts as a usable type.
    entry: { index: "./src/index.ts", nn: "./src/nn/index.ts" },
    platform: "node",
    dts: true,
    sourcemap: true,
    clean: true,
    // type:module → emit .js/.d.ts (ESM) consistently. Without this, the node
    // platform defaults to fixedExtension=true (.mjs/.d.mts), mismatching the
    // package.json "exports" paths and the browser config's .js output.
    fixedExtension: false,
  },
  {
    entry: { browser: "./src/browser.ts" },
    platform: "browser",
    dts: true,
    // minify disabled: esbuild converts Math.pow(await x, y) → (await x)**y
    // which is a SyntaxError in browsers (unary before exponentiation)
    minify: false,
    sourcemap: true,
    // Do NOT clean: the first (node) config already cleaned dist; cleaning again
    // here would wipe the node/nn outputs (tsdown cleans per-config by default).
    clean: false,
  },
]);
