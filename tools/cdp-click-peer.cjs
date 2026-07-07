// Connects to the user's real Mac Chrome over the forwarded CDP port (9222),
// finds the already-open /pretrain-v2 tab, streams its console, and clicks
// "join + train". Drives the user's actual browser/GPU as the DiLoCo peer.
// Does NOT launch or close the browser — it's the user's.
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.connectOverCDP("http://localhost:9222");
  const contexts = browser.contexts();
  let page = null;
  for (const ctx of contexts) {
    for (const p of ctx.pages()) {
      if (p.url().includes("/pretrain-v2")) { page = p; break; }
    }
    if (page) break;
  }
  if (!page) { console.error("no /pretrain-v2 tab found"); process.exit(1); }
  console.log("attached to tab:", page.url());

  page.on("console", (msg) => {
    const t = msg.text();
    if (t.includes("pretrain-v2") || t.includes("STATS") || /error|fail|nan/i.test(t)) {
      console.log(new Date().toISOString().slice(11, 19), t);
    }
  });
  page.on("pageerror", (e) => console.log("PAGEERROR", e.message));

  const gpu = await page.evaluate(async () => {
    if (!("gpu" in navigator)) return "no navigator.gpu";
    try { const a = await navigator.gpu.requestAdapter(); return a ? "adapter OK" : "no adapter"; }
    catch (e) { return "adapter err: " + e.message; }
  });
  console.log("WebGPU:", gpu);

  const btn = page.locator('button:has-text("join + train")');
  const running = await page.locator('button:has-text("running")').count();
  if (running > 0) {
    console.log("already running — just streaming console");
  } else {
    await btn.waitFor({ timeout: 10000 });
    await btn.click();
    console.log("clicked join + train on your Mac Chrome — streaming…");
  }

  const soakMs = parseInt(process.env.SOAK_MS || "1800000", 10);
  await new Promise((r) => setTimeout(r, soakMs));
  // Detach without closing the user's browser.
  process.exit(0);
})().catch((e) => { console.error("CDP DRIVER FATAL", e.message); process.exit(1); });
