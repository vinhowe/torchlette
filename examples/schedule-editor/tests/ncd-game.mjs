import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import process from "node:process";
import { chromium } from "playwright";

const PORT = 43180;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const server = spawn(
  "pnpm",
  ["exec", "vite", "preview", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"],
  { cwd: process.cwd(), detached: true, stdio: "ignore" },
);

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      if ((await fetch(ORIGIN)).ok) return;
    } catch {
      // Preview is still binding.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Timed out waiting for the NCD learning-game preview");
}

async function capture(page, name) {
  await page.screenshot({ path: `review/checkpoint-f-${name}.png` });
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(ORIGIN);
  await page.getByRole("button", { name: "NCD diagram" }).click();
  await page.addStyleTag({ content: "*{animation:none!important;transition:none!important}" });

  // Level 0 begins with an action, not a menu or jargon.
  assert.equal(await page.getByTestId("traffic-value").innerText(), "64 MB");
  assert.equal(await page.getByText("Hₗ₁").count(), 0);
  await capture(page, "level0-goal");
  const introParcel = page.getByRole("button", { name: /Temporary result parcel/ });
  const introTarget = page.locator('[data-nearby-drop="true"]');
  const parcelBox = await introParcel.boundingBox();
  const targetBox = await introTarget.boundingBox();
  assert.ok(parcelBox && targetBox);
  await page.mouse.move(parcelBox.x + parcelBox.width / 2, parcelBox.y + parcelBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(targetBox.x + targetBox.width / 2, targetBox.y + targetBox.height / 2 + 70, { steps: 6 });
  await capture(page, "level0-first-action");
  await page.mouse.move(targetBox.x + targetBox.width / 2, targetBox.y + targetBox.height / 2, { steps: 3 });
  await page.mouse.up();
  assert.equal(await page.getByTestId("traffic-value").innerText(), "48 MB");
  assert.match(await page.getByTestId("interpretation").innerText(), /removed one round trip/i);
  assert.match(await page.getByTestId("interpretation").innerText(), /16 MB/);
  await capture(page, "level0-aha");
  await page.getByRole("button", { name: "Show me the shorthand" }).click();
  assert.match(await page.getByTestId("notation-reveal").innerText(), /keep this value nearby/i);
  await capture(page, "level0-completion");
  await page.getByRole("button", { name: /Continue to the chain/ }).click();
  await page.getByTestId("lesson-map").waitFor();
  console.log("PASS level 0: act → physical consequence → earned notation");

  // Level 1 repeats the direct manipulation twice and names fusion after mastery.
  await page.getByRole("button", { name: /Open Fuse the chain/ }).click();
  assert.match(await page.getByTestId("teaching-feedback").innerText(), /temporary parcels/i);
  await capture(page, "level1-goal");
  await page.getByTestId("boundary-a").click();
  assert.match(await page.getByTestId("teaching-feedback").innerText(), /16 MB less traffic/i);
  await capture(page, "level1-first-action");
  await page.getByTestId("boundary-b").click();
  assert.match(await page.getByTestId("level-completion").innerText(), /That is fusion/i);
  await capture(page, "level1-aha");
  await page.getByRole("button", { name: /Back to lesson map/ }).last().click();
  assert.match(await page.getByRole("button", { name: /Open Fuse the chain/ }).innerText(), /✓/);
  await capture(page, "level1-completion");
  console.log("PASS level 1: two concrete shortcuts → fusion named at completion");

  // LayerNorm: the failed prediction exposes the dependency; Welford is operated before named.
  await page.getByRole("button", { name: /Open Carry the moments/ }).click();
  await capture(page, "level3-goal");
  await page.getByRole("button", { name: /Try one continuous pass/ }).click();
  assert.match(await page.getByTestId("layernorm-failure").innerText(), /mean so far[\s\S]*becomes/i);
  await capture(page, "level3-first-action");
  await page.getByRole("button", { name: /4-number experiment/ }).click({ force: true });
  assert.equal(await page.getByTestId("welford-lab").getByText("Welford", { exact: false }).count(), 0);
  await page.getByRole("button", { name: /Feed \[2, 4\]/ }).click({ force: true });
  await page.getByRole("button", { name: /Feed \[8, 10\]/ }).click({ force: true });
  assert.match(await page.getByTestId("welford-lab").innerText(), /count[\s\S]*4[\s\S]*mean[\s\S]*6[\s\S]*M2[\s\S]*40/i);
  await capture(page, "level3-aha");
  await page.getByRole("button", { name: /Carry this backpack/ }).click({ force: true });
  assert.match(await page.locator(".earned-tool").innerText(), /Welford running moments/);
  await page.getByRole("button", { name: /Keep each chunk nearby/ }).click({ force: true });
  await page.getByRole("button", { name: /Flow 128 values/ }).click({ force: true });
  assert.match(await page.getByTestId("level-completion").innerText(), /three-number backpack/i);
  await capture(page, "level3-completion");
  await page.getByRole("button", { name: /Back to lesson map/ }).last().click();
  console.log("PASS level 3: productive failure → operated summary → Welford → flowing row");

  // Softmax: a wrong choice remains visible, then the player repairs the moving scale.
  await page.getByRole("button", { name: /Open Cross the lemma wall/ }).click();
  await capture(page, "level8-goal");
  await page.getByRole("button", { name: /Try one continuous pass/ }).click();
  assert.match(await page.getByTestId("softmax-failure").innerText(), /later maximum moves the ruler/i);
  await capture(page, "level8-first-action");
  await page.getByRole("button", { name: /Try it with/ }).click({ force: true });
  assert.equal(await page.getByTestId("softmax-lab").getByText("online softmax", { exact: false }).count(), 0);
  await page.getByRole("button", { name: /Reveal first block/ }).click({ force: true });
  await page.getByRole("button", { name: /Reveal next block/ }).click({ force: true });
  await page.getByRole("button", { name: /Keep 1.368 and add/ }).click({ force: true });
  assert.match(await page.getByTestId("teaching-feedback").innerText(), /would not normalize/i);
  await page.getByRole("button", { name: /Rescale the old subtotal/ }).click({ force: true });
  assert.match(await page.getByTestId("softmax-correction").innerText(), /exp\(2 − 4\)/);
  await capture(page, "level8-aha");
  await page.getByRole("button", { name: /Carry the ruler/ }).click({ force: true });
  assert.match(await page.locator(".earned-tool").innerText(), /Online softmax/);
  await page.getByRole("button", { name: /Keep score chunks nearby/ }).click();
  await page.getByRole("button", { name: /Flow 128 scores/ }).click();
  assert.match(await page.getByTestId("level-completion").innerText(), /repairs the old subtotal/i);
  await capture(page, "level8-completion");
  await page.getByRole("button", { name: /Back to lesson map/ }).last().click();
  console.log("PASS level 8: moving-ruler counterexample → rescaling insight → online softmax");

  // Capstone: no new mechanics. A premature stream points back to the learned dependency.
  await page.getByRole("button", { name: /Open Assemble FlashAttention/ }).click();
  await capture(page, "level9-goal");
  await page.getByRole("button", { name: /Flow 32 key columns/ }).click();
  assert.match(await page.getByTestId("teaching-feedback").innerText(), /Reuse the m,ℓ backpack first/i);
  await capture(page, "level9-first-action");
  await page.getByRole("button", { name: /Carry m and ℓ/ }).click();
  await page.getByRole("button", { name: /Keep score blocks nearby/ }).click();
  await page.getByRole("button", { name: /Keep probability blocks nearby/ }).click();
  await capture(page, "level9-aha");
  await page.getByRole("button", { name: /Work on 64 query rows/ }).click();
  await page.getByRole("button", { name: /Flow 32 key columns/ }).click();
  assert.match(await page.getByTestId("level-completion").innerText(), /built the FlashAttention schedule/i);
  assert.match(await page.getByTestId("level-completion").innerText(), /54 MB[\s\S]*576 KB/i);
  await capture(page, "level9-completion");
  await page.getByRole("button", { name: /Read the paper shorthand/ }).click();
  await page.getByTestId("earned-notation").waitFor();
  assert.match(await page.getByTestId("earned-notation").innerText(), /warm regions stayed nearby/i);
  console.log("PASS level 9: four earned ideas compose into FlashAttention; notation arrives last");

  assert.deepEqual(errors, []);
  console.log("NCD learning-game acceptance: level 0 + 4/4 lesson paths passed");
} finally {
  await browser?.close();
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    // Preview may already have exited after a failed assertion.
  }
}
