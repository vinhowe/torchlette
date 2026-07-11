import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import process from "node:process";
import { chromium } from "playwright";

const PORT = 43179;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const server = spawn(
  "pnpm",
  [
    "exec",
    "vite",
    "preview",
    "--host",
    "127.0.0.1",
    "--port",
    String(PORT),
    "--strictPort",
  ],
  { cwd: process.cwd(), detached: true, stdio: "ignore" },
);

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(ORIGIN);
      if (response.ok) return;
    } catch {
      // The preview process is still binding.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Timed out waiting for the NCD interaction-test preview");
}

function nearlyEqual(actual, expected, tolerance = 0.25) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1100 },
  });
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  async function openSurface() {
    await page.goto(ORIGIN);
    await page.getByRole("button", { name: "NCD diagram" }).click();
    await page.getByRole("button", { name: "Open freeform sandbox" }).click();
    const viewport = page.locator(".ncd-viewport");
    await viewport.waitFor();
    return viewport;
  }

  async function transform(viewport) {
    return viewport.evaluate((element) => ({
      panX: Number(element.dataset.panX),
      panY: Number(element.dataset.panY),
      scale: Number(element.dataset.scale),
      css: element.querySelector(".ncd-world")?.getAttribute("style") ?? "",
    }));
  }

  async function settleFrames(count = 3) {
    await page.evaluate(
      (frames) =>
        new Promise((resolve) => {
          const step = () => {
            frames -= 1;
            if (frames <= 0) resolve();
            else requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }),
      count,
    );
  }

  // (a) Ordinary two-finger/wheel input pans and cannot change scale.
  let viewport = await openSurface();
  let box = await viewport.boundingBox();
  assert.ok(box);
  const wheelBefore = await transform(viewport);
  await page.mouse.move(box.x + 900, box.y + 400);
  await page.mouse.wheel(24, 42);
  await settleFrames();
  const wheelAfter = await transform(viewport);
  assert.equal(wheelAfter.scale, wheelBefore.scale);
  assert.notEqual(wheelAfter.panX, wheelBefore.panX);
  assert.notEqual(wheelAfter.panY, wheelBefore.panY);
  console.log("PASS wheel without ctrl pans; scale unchanged");

  // (b) Modifier-wheel zoom keeps the pointer's world coordinate invariant.
  await page.reload();
  viewport = await openSurface();
  box = await viewport.boundingBox();
  assert.ok(box);
  const pointer = { x: box.x + 640, y: box.y + 330 };
  const zoomBefore = await transform(viewport);
  const beforeWorld = {
    x: (pointer.x - box.x - zoomBefore.panX) / zoomBefore.scale,
    y: (pointer.y - box.y - zoomBefore.panY) / zoomBefore.scale,
  };
  await page.mouse.move(pointer.x, pointer.y);
  await page.keyboard.down("Control");
  await page.mouse.wheel(0, -25);
  await page.keyboard.up("Control");
  await settleFrames();
  const zoomAfter = await transform(viewport);
  const afterWorld = {
    x: (pointer.x - box.x - zoomAfter.panX) / zoomAfter.scale,
    y: (pointer.y - box.y - zoomAfter.panY) / zoomAfter.scale,
  };
  assert.ok(zoomAfter.scale > zoomBefore.scale);
  nearlyEqual(afterWorld.x, beforeWorld.x);
  nearlyEqual(afterWorld.y, beforeWorld.y);
  console.log("PASS ctrl+wheel zooms about the pointer");

  // Ten gentle pinch ticks should be approximately a 2× zoom, not an explosion.
  await page.reload();
  viewport = await openSurface();
  box = await viewport.boundingBox();
  assert.ok(box);
  const gentleBefore = await transform(viewport);
  await page.mouse.move(box.x + 700, box.y + 360);
  for (let index = 0; index < 10; index += 1) {
    await page.keyboard.down("Control");
    await page.mouse.wheel(0, -25);
    await page.keyboard.up("Control");
    await settleFrames(2);
  }
  const gentleAfter = await transform(viewport);
  assert.ok(gentleAfter.scale / gentleBefore.scale > 1.8);
  assert.ok(gentleAfter.scale / gentleBefore.scale < 2.2);
  console.log("PASS ten gentle pinch ticks are approximately 2×");

  // (c) Empty-space primary-button drag pans with no scale mutation.
  await page.reload();
  viewport = await openSurface();
  box = await viewport.boundingBox();
  assert.ok(box);
  const dragBefore = await transform(viewport);
  await page.mouse.move(box.x + box.width - 110, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 50, box.y + box.height / 2 + 35, {
    steps: 4,
  });
  await settleFrames();
  await page.mouse.up();
  const dragAfter = await transform(viewport);
  assert.equal(dragAfter.scale, dragBefore.scale);
  assert.ok(dragAfter.panX - dragBefore.panX > 50);
  assert.ok(dragAfter.panY - dragBefore.panY > 25);
  console.log("PASS empty-space pointer drag pans");

  // (d) A drag over mathematical text cannot create a document selection.
  const heading = page.locator(".ncd-cost-heading").nth(2);
  const headingBox = await heading.boundingBox();
  assert.ok(headingBox);
  await page.mouse.move(headingBox.x + 4, headingBox.y + headingBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    headingBox.x + headingBox.width - 4,
    headingBox.y + headingBox.height / 2,
    { steps: 5 },
  );
  await page.mouse.up();
  assert.equal(
    await page.evaluate(() => window.getSelection()?.isCollapsed),
    true,
  );
  console.log("PASS canvas drag produces no document selection");

  // (e) Escape during a captured pan restores the exact prior transform.
  const escapeBefore = await transform(viewport);
  box = await viewport.boundingBox();
  assert.ok(box);
  await page.mouse.move(box.x + box.width - 120, box.y + box.height / 2 + 80);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 45, box.y + box.height / 2 + 120);
  await settleFrames();
  const escapeMid = await transform(viewport);
  assert.notEqual(escapeMid.panX, escapeBefore.panX);
  await page.keyboard.press("Escape");
  await settleFrames();
  const escapeAfter = await transform(viewport);
  await page.mouse.up();
  nearlyEqual(escapeAfter.panX, escapeBefore.panX, 1e-6);
  nearlyEqual(escapeAfter.panY, escapeBefore.panY, 1e-6);
  nearlyEqual(escapeAfter.scale, escapeBefore.scale, 1e-6);
  console.log("PASS Escape restores the pre-drag transform");

  // (f) Fifty same-frame wheel events coalesce to one finite, clamped transform.
  box = await viewport.boundingBox();
  assert.ok(box);
  await viewport.evaluate(
    (element, point) => {
      for (let index = 0; index < 50; index += 1) {
        element.dispatchEvent(
          new WheelEvent("wheel", {
            bubbles: true,
            cancelable: true,
            clientX: point.x,
            clientY: point.y,
            ctrlKey: true,
            deltaY: -120,
          }),
        );
      }
    },
    { x: box.x + 500, y: box.y + 300 },
  );
  await settleFrames(4);
  const spamAfter = await transform(viewport);
  assert.ok(Number.isFinite(spamAfter.panX));
  assert.ok(Number.isFinite(spamAfter.panY));
  assert.ok(Number.isFinite(spamAfter.scale));
  assert.ok(spamAfter.scale >= 0.36 && spamAfter.scale <= 2.5);
  assert.doesNotMatch(spamAfter.css, /NaN|Infinity/);
  console.log(
    "PASS 50 rapid wheel events produce a bounded coherent transform",
  );

  // Shift-wheel follows the desktop-canvas convention of horizontal-only pan.
  await page.reload();
  viewport = await openSurface();
  box = await viewport.boundingBox();
  assert.ok(box);
  const shiftBefore = await transform(viewport);
  await page.mouse.move(box.x + 800, box.y + 350);
  await page.keyboard.down("Shift");
  await page.mouse.wheel(0, 48);
  await page.keyboard.up("Shift");
  await settleFrames();
  const shiftAfter = await transform(viewport);
  assert.notEqual(shiftAfter.panX, shiftBefore.panX);
  assert.equal(shiftAfter.panY, shiftBefore.panY);
  console.log("PASS shift+wheel pans horizontally only");

  // Resize keeps the same world coordinate at the viewport center.
  box = await viewport.boundingBox();
  assert.ok(box);
  const resizeBefore = await transform(viewport);
  const centerBefore = {
    x: (box.width / 2 - resizeBefore.panX) / resizeBefore.scale,
    y: (box.height / 2 - resizeBefore.panY) / resizeBefore.scale,
  };
  await page.setViewportSize({ width: 1400, height: 1100 });
  await settleFrames(4);
  const resizedBox = await viewport.boundingBox();
  assert.ok(resizedBox);
  const resizeAfter = await transform(viewport);
  nearlyEqual(
    (resizedBox.width / 2 - resizeAfter.panX) / resizeAfter.scale,
    centerBefore.x,
  );
  nearlyEqual(
    (resizedBox.height / 2 - resizeAfter.panY) / resizeAfter.scale,
    centerBefore.y,
  );
  console.log("PASS resize preserves the world-space view center");

  // Escape also cancels an armed paint before pointer release.
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.reload();
  viewport = await openSurface();
  await page.getByRole("button", { name: "ℓ0 · Global" }).click();
  const paintTarget = page.locator('[data-wire="scores"][data-column="3"]');
  const paintBefore = await paintTarget.getAttribute("aria-label");
  const paintBox = await paintTarget.boundingBox();
  assert.ok(paintBox);
  await page.mouse.move(
    paintBox.x + paintBox.width / 2,
    paintBox.y + paintBox.height / 2,
  );
  await page.mouse.down();
  await page.keyboard.press("Escape");
  await page.mouse.up();
  assert.equal(await paintTarget.getAttribute("aria-label"), paintBefore);
  console.log("PASS Escape cancels an in-flight paint without committing");

  // Palette drags ending without a drop clear every target affordance.
  const groupChip = page.getByRole("button", { name: "gₐ · Group" });
  await groupChip.evaluate((element) => {
    const dataTransfer = new DataTransfer();
    element.dispatchEvent(
      new DragEvent("dragstart", {
        bubbles: true,
        cancelable: true,
        dataTransfer,
      }),
    );
  });
  assert.ok((await page.locator(".drop-valid").count()) > 0);
  await groupChip.dispatchEvent("dragend");
  assert.equal(await page.locator(".drop-valid").count(), 0);
  console.log("PASS drag release outside the canvas cancels target state");

  // Empty-canvas double-click is deliberately inert.
  box = await viewport.boundingBox();
  assert.ok(box);
  const doubleBefore = await transform(viewport);
  await page.mouse.dblclick(box.x + box.width - 100, box.y + box.height / 2);
  await settleFrames();
  const doubleAfter = await transform(viewport);
  nearlyEqual(doubleAfter.panX, doubleBefore.panX, 1e-6);
  nearlyEqual(doubleAfter.panY, doubleBefore.panY, 1e-6);
  nearlyEqual(doubleAfter.scale, doubleBefore.scale, 1e-6);
  console.log("PASS empty-canvas double-click is inert");

  assert.deepEqual(consoleErrors, []);
  console.log("NCD aggressive-user protocol: 12/12 assertions passed");
} finally {
  await browser?.close();
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    // The preview may already have exited after a failed assertion.
  }
}
