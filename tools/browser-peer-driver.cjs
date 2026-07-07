// Drives the bundled Playwright Chromium as a real browser DiLoCo peer against
// the local vite /pretrain-v2 page. Same engine + WebGPU flags as the project's
// browser tests. Streams the page console so we can watch the mesh form.
const { chromium } = require("playwright");

const URL =
  process.env.PEER_URL ||
  "http://localhost:5173/pretrain-v2?server=ws://5.78.181.14:443&layers=12&embed=768&heads=12";

(async () => {
  const browser = await chromium.launch({
    headless: true,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--no-sandbox",
    ],
  });
  const ctx = await browser.newContext();
  const page = await ctx.newPage();

  page.on("console", (msg) => {
    const t = msg.text();
    // The page logs everything through console.log("[pretrain-v2]", ...).
    if (t.includes("pretrain-v2") || t.includes("STATS") || /error|fail|nan/i.test(t)) {
      console.log(new Date().toISOString().slice(11, 19), t);
    }
  });
  page.on("pageerror", (e) => console.log("PAGEERROR", e.message));

  console.log("navigating to", URL);
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });

  // Confirm WebGPU is actually present in this Chromium before we try to train.
  const gpu = await page.evaluate(async () => {
    if (!("gpu" in navigator)) return "no navigator.gpu";
    try {
      const a = await navigator.gpu.requestAdapter();
      return a ? "adapter OK" : "no adapter";
    } catch (e) {
      return "adapter err: " + e.message;
    }
  });
  console.log("WebGPU:", gpu, "| crossOriginIsolated:", await page.evaluate(() => globalThis.crossOriginIsolated));

  const btn = page.locator('button:has-text("join + train")');
  await btn.waitFor({ timeout: 15000 });
  await btn.click();
  console.log("clicked join+train — training…");

  // Stay alive; the console handler streams progress. Exit after a long soak
  // or when killed.
  const soakMs = parseInt(process.env.SOAK_MS || "1800000", 10);
  await page.waitForTimeout(soakMs);
  await browser.close();
  process.exit(0);
})().catch((e) => {
  console.error("DRIVER FATAL", e);
  process.exit(1);
});
