import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/browser/**/*.spec.ts"],
    browser: {
      enabled: true,
      provider: "playwright",
      instances: [
        {
          browser: "chromium",
          launch: {
            // WebGPU requires these flags in headless mode
            args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
          },
        },
      ],
    },
    // Longer timeout for GPU operations
    testTimeout: 30000,
  },
});
