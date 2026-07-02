import { chromium } from "playwright";
async function main() {
  const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--use-angle=vulkan", "--ignore-gpu-blocklist", "--use-vulkan=native"] });
  const page = await browser.newPage();
  await page.goto("http://localhost:5173/");
  const info = await page.evaluate(async () => {
    const a = await (navigator as any).gpu?.requestAdapter();
    if (!a) return "no adapter";
    const i = a.info ?? {};
    return { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, f16: a.features.has("shader-f16") };
  });
  console.log(JSON.stringify(info));
  await browser.close();
  process.exit(0);
}
main();
