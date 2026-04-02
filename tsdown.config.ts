import { defineConfig } from "tsdown";

export default defineConfig([
  {
    entry: ["./src/index.ts"],
    platform: "node",
    dts: true,
    sourcemap: true,
  },
  {
    entry: ["./bench/runner.ts"],
    platform: "node",
    dts: false,
    sourcemap: true,
    outDir: "dist/bench",
  },
  {
    entry: ["./src/browser.ts"],
    platform: "browser",
    dts: true,
    // minify disabled: esbuild converts Math.pow(await x, y) → (await x)**y
    // which is a SyntaxError in browsers (unary before exponentiation)
    minify: false,
    sourcemap: true,
  },
]);
