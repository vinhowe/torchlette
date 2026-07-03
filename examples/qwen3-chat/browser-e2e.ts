/**
 * Headless end-to-end check of the qwen3-chat app's IN-BROWSER mode on this
 * box (Chromium + Vulkan WebGPU on the V100). Loads the 0.6B model from HF
 * into the tab, sends a chat message, verifies a streamed reply.
 *
 * Run from repo root: npx tsx examples/qwen3-chat/browser-e2e.ts [ui|full]
 */

import { chromium } from "playwright";

const MODE = process.argv[2] ?? "ui";
const URL = "http://localhost:5173/";
const SHOT_DIR = process.env.SHOT_DIR ?? "/tmp";

async function main() {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--use-angle=vulkan", "--ignore-gpu-blocklist", "--use-vulkan=native"],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));

  await page.goto(URL, { waitUntil: "networkidle" });
  await page.screenshot({ path: `${SHOT_DIR}/chat-light.png` });
  // Toggle dark via the ThemeToggle's Dark option if present, else class hack.
  await page.evaluate(() => document.documentElement.classList.add("dark"));
  await page.screenshot({ path: `${SHOT_DIR}/chat-dark.png` });
  await page.evaluate(() => document.documentElement.classList.remove("dark"));

  const webgpu = await page.evaluate(async () => {
    if (!("gpu" in navigator)) return "no navigator.gpu";
    const adapter = await (navigator as any).gpu.requestAdapter();
    if (!adapter) return "no adapter";
    return `adapter ok, f16=${adapter.features.has("shader-f16")}, maxBuffer=${adapter.limits.maxBufferSize}`;
  });
  console.log("webgpu:", webgpu);
  console.log("console errors after load:", JSON.stringify(consoleErrors));

  if (MODE === "full") {
    await page.getByRole("button", { name: "In-browser" }).click();
    await page.getByRole("button", { name: "0.6B", exact: true }).click();
    await page.getByRole("button", { name: "Load model" }).click();
    console.log("loading 0.6B from HF… (this downloads ~1.5GB)");
    // Wait until the engine badge appears (model resident) — up to 15 min.
    // Poll so we can surface in-page errors and progress along the way.
    const deadline = Date.now() + 15 * 60 * 1000;
    let lastStatus = "";
    for (;;) {
      if (Date.now() > deadline) {
        await page.screenshot({ path: `${SHOT_DIR}/chat-timeout.png` });
        console.log("PAGE TEXT:", await page.evaluate(() => document.body.innerText.slice(0, 2000)));
        console.log("CONSOLE ERRORS:", JSON.stringify(consoleErrors.slice(-10)));
        throw new Error("timed out waiting for model resident");
      }
      const state = await page.evaluate(() => ({
        resident: document.body.innerText.includes("resident"),
        error: (document.body.innerText.match(/Error[\s\S]{0,300}/) || [null])[0],
        status: (document.body.innerText.match(/Loading weights…[^\n]*|Fetching[^\n]*|Loading tokenizer[^\n]*/) || [""])[0],
      }));
      if (state.error) {
        await page.screenshot({ path: `${SHOT_DIR}/chat-error.png` });
        console.log("CONSOLE ERRORS:", JSON.stringify(consoleErrors.slice(-10)));
        throw new Error(`in-page error: ${state.error}`);
      }
      if (state.status && state.status !== lastStatus) {
        lastStatus = state.status;
        console.log("progress:", state.status);
      }
      if (state.resident) break;
      await new Promise((r) => setTimeout(r, 5000));
    }
    console.log("model resident");
    await page.screenshot({ path: `${SHOT_DIR}/chat-loaded.png` });

    await page.locator("textarea").fill("Reply with exactly: BROWSER OK");
    await page.getByRole("button", { name: "Send" }).click();
    // Wait for a non-empty assistant message and generation to finish (Send re-enabled).
    await page.waitForFunction(
      () => {
        const ps = Array.from(document.querySelectorAll("main p"));
        return ps.length >= 2 && (ps[ps.length - 1].textContent ?? "").trim().length > 0;
      },
      undefined,
      { timeout: 5 * 60 * 1000 },
    );
    // Generation finished ⇔ busy=false ⇔ the textarea re-enables. (The Send
    // button stays disabled on empty input, so it can't signal completion.)
    await page.waitForFunction(
      () => {
        const ta = document.querySelector("textarea");
        const err = document.body.innerText.match(/Error[\s\S]{0,200}/);
        if (err) throw new Error("in-page: " + err[0]);
        return ta && !ta.disabled;
      },
      undefined,
      { timeout: 5 * 60 * 1000 },
    );
    const reply = await page.evaluate(() => {
      const ps = Array.from(document.querySelectorAll("main p"));
      return ps[ps.length - 1].textContent;
    });
    const statsText = await page.evaluate(() =>
      Array.from(document.querySelectorAll("span"))
        .filter((s) => s.className.includes("type-value"))
        .map((s) => s.textContent)
        .join(" | "),
    );
    console.log("assistant reply:", JSON.stringify(reply));
    console.log("stats:", statsText);
    await page.screenshot({ path: `${SHOT_DIR}/chat-replied.png` });
    console.log("recent console errors:", JSON.stringify(consoleErrors.slice(-5)));

    // --- Warm-cache phase: reload the page (same profile/origin → IndexedDB
    // persists) and load again; must come from cache and be much faster.
    console.log("reloading for warm-cache load…");
    await page.reload({ waitUntil: "networkidle" });
    await page.getByRole("button", { name: "In-browser" }).click();
    await page.getByRole("button", { name: "0.6B", exact: true }).click();
    const tCache = Date.now();
    await page.getByRole("button", { name: "Load model" }).click();
    let sawCached = false;
    for (;;) {
      if (Date.now() - tCache > 5 * 60 * 1000) throw new Error("warm-cache load timed out");
      const state = await page.evaluate(() => ({
        resident: document.body.innerText.includes("resident"),
        cached: document.body.innerText.includes("(cached)"),
        error: (document.body.innerText.match(/Error[\s\S]{0,200}/) || [null])[0],
      }));
      if (state.error) throw new Error(`warm-cache in-page error: ${state.error}`);
      sawCached ||= state.cached;
      if (state.resident) break;
      await new Promise((r) => setTimeout(r, 1000));
    }
    console.log(`warm-cache load: ${((Date.now() - tCache) / 1000).toFixed(1)}s, sawCached=${sawCached}`);
    if (!sawCached) throw new Error("warm load did not use the IndexedDB cache");

  }

  await browser.close();
  console.log("E2E DONE");
  process.exit(0);
}

main().catch((e) => {
  console.error("E2E FAILED:", e);
  process.exit(1);
});
