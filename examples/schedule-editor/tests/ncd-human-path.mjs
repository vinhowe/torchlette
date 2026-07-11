import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import process from "node:process";
import { chromium } from "playwright";

const PORT = 43184;
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
  throw new Error("Timed out waiting for the human-path preview");
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  function marked(actionId) {
    return page.locator(`[data-game-affordance][data-action-id="${actionId}"]`);
  }

  async function bringIntoView(locator) {
    for (let attempt = 0; attempt < 20; attempt += 1) {
      const box = await locator.boundingBox();
      if (box && box.y >= 0 && box.y + Math.min(box.height, 40) <= 990) return box;
      await page.mouse.move(720, 850);
      await page.mouse.wheel(0, box && box.y < 68 ? -520 : 520);
      await page.waitForTimeout(40);
    }
    throw new Error("Marked affordance never entered the viewport");
  }

  async function pointerClick(actionId) {
    const locator = marked(actionId);
    assert.equal(await locator.count(), 1, `${actionId} must identify one marked affordance`);
    const box = await bringIntoView(locator);
    assert.equal(await locator.isEnabled(), true, `${actionId} must be enabled`);
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(80);
  }

  async function currentGuidance(expectedAction) {
    const guide = page.getByTestId("lesson-guidance");
    await guide.waitFor();
    assert.equal(await guide.count(), 1, "exactly one guidance layer must be mounted");
    assert.equal(await guide.getAttribute("data-target-action"), expectedAction);
    const target = marked(expectedAction);
    assert.equal(await target.getAttribute("data-current-target"), "true");
    assert.ok((await guide.innerText()).match(/click|drag/i), "guidance must state a concrete pointer gesture");
  }

  async function clickGuided(expectedAction) {
    await currentGuidance(expectedAction);
    await pointerClick(expectedAction);
  }

  async function capture(name) {
    await page.screenshot({ path: `review/checkpoint-g-${name}.png` });
  }

  async function assertLessonEntry(levelName, expectedAction, screenshot) {
    await pointerClick(levelName);
    assert.equal(
      await page.getByTestId("learning-game").evaluate((element) => element.scrollTop),
      0,
      "lesson entry must reset the shared scroll container",
    );
    await currentGuidance(expectedAction);
    // Chromium needs two compositing frames after the fixed scroll-world swaps
    // a long map for a lesson; without this, review screenshots can contain
    // transient black tiles even though the DOM and pointer target are ready.
    await page.waitForTimeout(2600);
    await capture(screenshot);
  }

  await page.goto(ORIGIN, { waitUntil: "networkidle" });
  await pointerClick("enter-game");

  // Level 0: only marked source/target elements receive pointer input.
  const introGuide = page.getByTestId("lesson-guidance");
  assert.equal(await introGuide.getAttribute("data-target-action"), "intro-drag");
  const parcel = marked("intro-drag");
  const shelf = marked("intro-nearby");
  assert.equal(await parcel.getAttribute("data-current-target"), "true");
  assert.equal(await shelf.getAttribute("data-current-target"), "true");
  await capture("level0-guidance");
  const parcelBox = await parcel.boundingBox();
  const shelfBox = await shelf.boundingBox();
  assert.ok(parcelBox && shelfBox);
  await page.mouse.move(parcelBox.x + parcelBox.width / 2, parcelBox.y + parcelBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(shelfBox.x + shelfBox.width / 2, shelfBox.y + shelfBox.height / 2, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(600);
  await pointerClick("intro-reveal");
  await pointerClick("intro-continue");
  await page.getByTestId("lesson-map").waitFor();
  console.log("PASS human path level 0: marked drag source → marked drop target");

  // Level 1: guidance names A then B; both gates look and behave as controls.
  await assertLessonEntry("open-fuse-chain", "fuse-boundary-a", "level1-guidance");
  await clickGuided("fuse-boundary-a");
  await clickGuided("fuse-boundary-b");
  await page.getByTestId("level-completion").waitFor();
  await pointerClick("lesson-map");
  console.log("PASS human path level 1: guidance mounted → two marked gates → completion");

  // Level 3: every phase advances through the currently named marked affordance.
  await assertLessonEntry("open-layernorm", "layernorm-try-flow", "level3-guidance");
  for (const action of [
    "layernorm-try-flow",
    "layernorm-open-lab",
    "welford-feed",
    "welford-feed",
    "welford-install",
    "layernorm-keep-nearby",
    "layernorm-stream",
  ]) await clickGuided(action);
  await page.getByTestId("level-completion").waitFor();
  await pointerClick("lesson-map");
  console.log("PASS human path level 3: 7 guided pointer actions → completion");

  // Level 8: choose the explicitly recommended repair path.
  await assertLessonEntry("open-softmax", "softmax-try-flow", "level8-guidance");
  for (const action of [
    "softmax-try-flow",
    "softmax-open-lab",
    "softmax-reveal",
    "softmax-reveal",
    "softmax-rescale",
    "softmax-install",
    "softmax-keep-nearby",
    "softmax-stream",
  ]) await clickGuided(action);
  await page.getByTestId("level-completion").waitFor();
  await pointerClick("lesson-map");
  console.log("PASS human path level 8: 8 guided pointer actions → completion");

  // Level 9: the guide names one valid start and sequences the reusable tools.
  await assertLessonEntry("open-attention", "fa-carry", "level9-guidance");
  for (const action of ["fa-carry", "fa-scores", "fa-probabilities", "fa-tile", "fa-stream"])
    await clickGuided(action);
  await page.getByTestId("level-completion").waitFor();
  console.log("PASS human path level 9: 5 visibly marked learned tools → completion");

  assert.deepEqual(pageErrors, []);
  console.log("NCD human-path acceptance: guidance 4/4, first actions 4/4, completions 4/4");
} finally {
  await browser?.close();
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    // Preview may already have exited after an assertion.
  }
}
