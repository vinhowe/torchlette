import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [
    tailwindcss(),
    svelte({
      onwarn(warning, handler) {
        // The verbatim registry ThemeToggle accepts a fixed `integrated` prop.
        if (warning.code === "state_referenced_locally") return;
        handler(warning);
      },
    }),
  ],
  resolve: {
    alias: {
      "$app/environment": path.resolve(here, "src/lib/app-environment.ts"),
    },
  },
});
