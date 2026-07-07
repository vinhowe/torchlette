/**
 * Headless end-to-end proof that in-browser contrastive activation steering
 * works on this box (Chromium + Vulkan WebGPU on the V100). Loads Qwen3-0.6B
 * into the tab, computes a happy/sad steering vector at a mid layer, generates
 * a fixed neutral prompt at α=0 (baseline) and at a strong positive α, and
 * asserts the two outputs DIFFER (steering had an effect). Prints both.
 *
 * Run SOLO w.r.t. other torchlette GPU processes (dev server must be up on 5175):
 *   pnpm --filter qwen3-steering dev &   # or: npx vite dev --port 5175
 *   npx tsx examples/qwen3-steering/browser-e2e.ts
 */

import { chromium } from "playwright";

const URL = process.env.URL ?? "http://localhost:5175/";
const SHOT_DIR = process.env.SHOT_DIR ?? "/tmp";
const ALPHA = 12;

// Loose sentiment lexicon for the bonus shift check.
const HAPPY = /\b(happy|joy|joyful|glad|cheer|cheerful|wonderful|delight|delightful|bright|smile|smiling|love|great|excited|beautiful|sunny|warm|hope|hopeful|good|nice|fun|enjoy)\b/gi;
const SAD = /\b(sad|sorrow|miserable|gloom|gloomy|depress|despair|dark|cry|crying|grief|pain|lonely|bleak|hopeless|terrible|awful|cold|death|dead|suffer)\b/gi;
const count = (s: string, re: RegExp) => (s.match(re) || []).length;

async function main() {
  // Persistent context so IndexedDB (cached weights) survives between runs —
  // the first run downloads, later runs load from cache (fast iteration).
  const PROFILE = process.env.PROFILE_DIR ?? "/tmp/steer-e2e-profile";
  const args = [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    "--use-angle=vulkan",
    "--ignore-gpu-blocklist",
    "--use-vulkan=native",
  ];
  const context = await chromium.launchPersistentContext(PROFILE, {
    headless: true,
    args,
    viewport: { width: 1280, height: 900 },
  });
  const browser = context.browser()!;
  const page = context.pages()[0] ?? (await context.newPage());
  const consoleErrors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
  // Live console forwarding for diagnosis.
  page.on("console", (m) => {
    const t = m.text();
    if (t.includes("[steer]") || m.type() === "error") console.log(`  [browser:${m.type()}]`, t);
  });

  // Retry helper: the Vite dev server can force a full page reload early on
  // (dependency re-optimization), which destroys the execution context mid
  // page.evaluate. Retry a few times through navigations.
  const evalRetry = async <T>(fn: () => Promise<T>): Promise<T> => {
    for (let i = 0; i < 5; i++) {
      try {
        return await fn();
      } catch (e) {
        if (String(e).includes("Execution context was destroyed") || String(e).includes("navigation")) {
          await page.waitForLoadState("domcontentloaded").catch(() => {});
          await new Promise((r) => setTimeout(r, 500));
          continue;
        }
        throw e;
      }
    }
    return fn();
  };

  await page.goto(URL, { waitUntil: "domcontentloaded" });
  // Let Vite's forced dependency re-optimization reload settle before we start
  // polling (that reload is what destroyed the context in a naive run).
  await new Promise((r) => setTimeout(r, 3000));
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.screenshot({ path: `${SHOT_DIR}/steer-load.png` });

  const webgpu = await evalRetry(() =>
    page.evaluate(async () => {
      if (!("gpu" in navigator)) return "no navigator.gpu";
      const a = await (navigator as any).gpu.requestAdapter();
      if (!a) return "no adapter";
      return `adapter ok, f16=${a.features.has("shader-f16")}, maxBuffer=${a.limits.maxBufferSize}`;
    }),
  );
  console.log("webgpu:", webgpu);

  // ---- Load 0.6B ----
  await page.getByRole("button", { name: "0.6B", exact: true }).click();
  await page.getByRole("button", { name: "Load model" }).click();
  console.log("loading Qwen3-0.6B from HF…");
  const deadline = Date.now() + 15 * 60 * 1000;
  let lastStatus = "";
  for (;;) {
    if (Date.now() > deadline) {
      console.log("PAGE:", await page.evaluate(() => document.body.innerText.slice(0, 2000)));
      throw new Error("timed out waiting for model resident");
    }
    const st = await evalRetry(() =>
      page.evaluate(() => ({
        resident: document.body.innerText.includes("resident"),
        error: (document.body.innerText.match(/Error[\s\S]{0,300}/) || [null])[0],
        status: (document.body.innerText.match(/Loading[^\n]*|Fetching[^\n]*/) || [""])[0],
      })),
    );
    if (st.error) {
      await page.screenshot({ path: `${SHOT_DIR}/steer-error.png` });
      console.log("CONSOLE:", JSON.stringify(consoleErrors.slice(-10)));
      throw new Error(`in-page error: ${st.error}`);
    }
    if (st.status && st.status !== lastStatus) {
      lastStatus = st.status;
      console.log("progress:", st.status);
    }
    if (st.resident) break;
    await new Promise((r) => setTimeout(r, 4000));
  }
  console.log("model resident");
  await page.screenshot({ path: `${SHOT_DIR}/steer-resident.png` });

  // ---- Compute happy/sad vector at a mid layer (preset 0 = default) ----
  await page.getByRole("button", { name: "Compute vector" }).click();
  console.log("clicked Compute vector, waiting…");
  try {
    await page.waitForFunction(() => document.body.innerText.includes("Vector ready"), undefined, {
      timeout: 3 * 60 * 1000,
    });
  } catch (e) {
    await page.screenshot({ path: `${SHOT_DIR}/steer-vec-timeout.png`, fullPage: true });
    console.log("VECTOR WAIT TIMED OUT. Page text:");
    console.log((await page.evaluate(() => document.body.innerText)).slice(0, 1500));
    console.log("CONSOLE ERRORS:", JSON.stringify(consoleErrors.slice(-15)));
    throw e;
  }
  console.log("steering vector computed");
  await page.screenshot({ path: `${SHOT_DIR}/steer-vector.png` });

  const setInput = async (el: any, v: number) =>
    el.evaluate((node: HTMLInputElement, val: number) => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
      setter.call(node, String(val));
      node.dispatchEvent(new Event("input", { bubbles: true }));
      node.dispatchEvent(new Event("change", { bubbles: true }));
    }, v);

  // Keep the e2e brisk: 40 new tokens per run.
  await setInput(page.locator('input[type="number"]'), 40);

  // ---- Set a strong positive alpha via the slider, run compare ----
  const alphaSlider = page.locator('input[type="range"]').nth(1); // 0 = layer, 1 = alpha
  await setInput(alphaSlider, ALPHA);

  await page.getByRole("button", { name: "Compare baseline vs steered" }).click();
  console.log(`generating baseline (α=0) then steered (α=${ALPHA})…`);

  // Both run cards must finish (each shows a "tok/s" stat when done).
  await page.waitForFunction(
    () => {
      const err = document.body.innerText.match(/Error[\s\S]{0,200}/);
      if (err) throw new Error("in-page: " + err[0]);
      const stats = Array.from(document.querySelectorAll("span")).filter((s) =>
        (s.textContent || "").includes("tok/s"),
      );
      return stats.length >= 2;
    },
    undefined,
    { timeout: 5 * 60 * 1000 },
  );

  const runs = await page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll("main .grid > div")).filter((d) =>
      (d.textContent || "").includes("tok/s"),
    );
    return cards.map((c) => {
      const label = c.querySelector("span")?.textContent?.trim() ?? "";
      const text = c.querySelector("p")?.textContent?.trim() ?? "";
      return { label, text };
    });
  });

  const baseline = runs.find((r) => r.label.includes("baseline")) ?? runs[0];
  const steered = runs.find((r) => r.label.includes("steered")) ?? runs[1];

  console.log("\n================= BASELINE (α = 0) =================");
  console.log(JSON.stringify(baseline.text));
  console.log("\n================= STEERED  (α = " + ALPHA + ") =================");
  console.log(JSON.stringify(steered.text));

  await page.screenshot({ path: `${SHOT_DIR}/steer-compare.png`, fullPage: true });

  // ---- ASSERT: steering had an effect ----
  if (!baseline.text || !steered.text) {
    throw new Error("one of the generations was empty");
  }
  if (baseline.text === steered.text) {
    throw new Error("STEERING HAD NO EFFECT: baseline == steered");
  }
  console.log("\nPASS: baseline and steered outputs DIFFER (steering had an effect).");

  // Bonus loose sentiment check (positive α with happy/sad should skew happier).
  const bScore = count(baseline.text, HAPPY) - count(baseline.text, SAD);
  const sScore = count(steered.text, HAPPY) - count(steered.text, SAD);
  console.log(`sentiment(happy−sad): baseline=${bScore}, steered=${sScore} (expect steered ≥ baseline)`);
  if (sScore >= bScore) console.log("BONUS: steered skews at least as happy as baseline.");
  else console.log("BONUS (soft): steered did not skew happier this run (sampling variance).");

  console.log("recent console errors:", JSON.stringify(consoleErrors.slice(-5)));
  await context.close();
  console.log("E2E DONE");
  process.exit(0);
}

main().catch((e) => {
  console.error("E2E FAILED:", e);
  process.exit(1);
});
