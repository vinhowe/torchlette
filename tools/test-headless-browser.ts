/**
 * Headless browser training test. Loads the demo, clicks Load Model,
 * then Train, and captures loss values from the UI. This tests the
 * EXACT same code path as what the user sees.
 */
import { chromium } from "playwright";

async function main() {
  console.log("Launching headless Chrome...");
  const browser = await chromium.launch({
    headless: true,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-vulkan",
      "--disable-vulkan-surface",
      "--ignore-gpu-blocklist",
      "--no-sandbox",
    ],
  });

  const page = await browser.newPage();
  const consoleLogs: string[] = [];
  page.on("console", (msg) => {
    const t = msg.text();
    consoleLogs.push(t);
  });
  page.on("pageerror", (err) =>
    console.log(`[PAGE ERROR] ${err.message.slice(0, 200)}`),
  );

  await page.goto("http://localhost:5173", {
    waitUntil: "networkidle",
    timeout: 30000,
  });
  await page.waitForTimeout(2000);
  console.log("Page loaded.");

  // Click "Load Model"
  console.log("Clicking Load Model...");
  await page.click('button:has-text("Load Model")');

  // Wait for model to load (up to 120s for weight download)
  console.log("Waiting for model to load...");
  for (let i = 0; i < 120; i++) {
    await page.waitForTimeout(1000);
    const text = await page.evaluate(() => document.body?.innerText || "");
    if (text.includes("Ready")) {
      console.log("Model loaded!");
      break;
    }
    if (text.includes("Error") || text.includes("error")) {
      console.log("Model load error:", text.slice(0, 300));
      await browser.close();
      process.exit(1);
    }
    if (i % 10 === 0) {
      // Find progress text
      const progress = text.match(/\d+%|Loading|Downloading|Checking/)?.[0];
      console.log(`  [${i}s] ${progress || "waiting..."}`);
    }
  }

  // Click "Train"
  console.log("Clicking Train...");
  await page.click('button:has-text("Train")');

  // Monitor loss values for 60 seconds
  const losses: number[] = [];
  console.log("Monitoring training...");
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(1000);
    const text = await page.evaluate(() => document.body?.innerText || "");

    // Extract step and loss from page text
    const stepMatch = text.match(/step\s*[:#]?\s*(\d+)/i);
    const lossMatch = text.match(/loss\s*[:#]?\s*([\d.]+)/i);
    if (stepMatch && lossMatch) {
      const step = parseInt(stepMatch[1]);
      const loss = parseFloat(lossMatch[1]);
      if (losses.length === 0 || step > losses.length - 1) {
        losses.push(loss);
        console.log(`  Step ${step}: loss=${loss.toFixed(4)}`);
      }
    }

    // Check if training finished
    if (losses.length >= 20) break;

    // Check for errors
    const errors = consoleLogs.filter(
      (l) =>
        l.includes("Error") ||
        l.includes("FAIL") ||
        l.includes("Input not ready"),
    );
    if (errors.length > 0) {
      console.log("Training errors found:");
      for (const e of errors.slice(0, 5)) console.log(`  ${e.slice(0, 200)}`);
      break;
    }
  }

  if (losses.length >= 2) {
    let increases = 0;
    for (let i = 1; i < losses.length; i++) {
      if (losses[i] > losses[i - 1] + 0.001) increases++;
    }
    console.log(`\nResults: ${losses.length} steps captured`);
    console.log(
      `  ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`,
    );
    console.log(`  Increases: ${increases}/${losses.length - 1}`);
    console.log(
      `  Trend: ${losses[losses.length - 1] < losses[0] ? "DECREASING" : "NOT DECREASING"}`,
    );
  } else {
    console.log("Not enough loss values captured. Console logs:");
    for (const l of consoleLogs.slice(-20)) console.log(`  ${l.slice(0, 200)}`);
  }

  await browser.close();
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e.message);
  process.exit(1);
});
