import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import process from "node:process";
import { chromium } from "playwright";

const PORT = 43180;
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
      if ((await fetch(ORIGIN)).ok) return;
    } catch {
      // Preview is still binding.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Timed out waiting for the NCD game preview");
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1100 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(ORIGIN);
  await page.getByRole("button", { name: "NCD diagram" }).click();
  await page.addStyleTag({
    content:
      "*{animation:none!important;transition:none!important}.ncd-equivalence{display:none!important}",
  });

  const capture = async (name) => {
    await page.waitForTimeout(250);
    await page.screenshot({ path: `review/checkpoint-e-${name}.png` });
  };

  async function chooseLevel(name) {
    await page.getByRole("button", { name }).click();
    await page.getByRole("button", { name: "Start level" }).waitFor();
  }

  async function startLevel() {
    await page.getByRole("button", { name: "Start level" }).click();
    await page.locator(".ncd-viewport").waitFor();
  }

  async function paint(wireId, column) {
    await page
      .locator(`[data-wire="${wireId}"][data-column="${column}"]`)
      .click({
        force: true,
      });
  }

  async function dropPartition(axisId, kind) {
    await page
      .locator(`[data-axis="${axisId}"]`)
      .first()
      .evaluate((element, partitionKind) => {
        const dataTransfer = new DataTransfer();
        dataTransfer.setData(
          "application/x-torchlette-ncd",
          JSON.stringify({ type: "partition", kind: partitionKind }),
        );
        element.dispatchEvent(
          new DragEvent("drop", {
            bubbles: true,
            cancelable: true,
            dataTransfer,
          }),
        );
      }, kind);
  }

  async function expectComplete(title, moves) {
    await page.getByText(/TARGET MET — LEVEL COMPLETE/).waitFor();
    assert.match(
      await page.locator("main").innerText(),
      new RegExp(`${moves} moves`),
    );
    await page.getByRole("button", { name: "Levels" }).click();
    const card = page.getByRole("button", { name: new RegExp(title) });
    assert.match(await card.innerText(), /COMPLETE/);
    assert.match(await card.innerText(), new RegExp(`Moves\\s+${moves}`));
  }

  // Exercise 1: the palette is exactly level colors + paint; two fusions win.
  await chooseLevel(/Fuse the chain/);
  await capture("level1-goal");
  await startLevel();
  assert.equal(
    await page.getByRole("button", { name: "gₐ · Group" }).count(),
    0,
  );
  assert.equal(
    await page.getByRole("button", { name: "sₐ · Stream" }).count(),
    0,
  );
  assert.equal(await page.getByRole("button", { name: /Lemma/ }).count(), 0);
  assert.deepEqual(
    await page.locator('button[draggable="true"]').allTextContents(),
    ["ℓ0 · Global", "ℓ1 · Lower"],
  );
  await capture("level1-jam-na-vocabulary");
  await paint("mid1", 2);
  await capture("level1-lemma-na-first-fusion");
  await paint("mid2", 4);
  await capture("level1-completion");
  await expectComplete("Fuse the chain", 2);
  console.log("PASS level 1: gated paint vocabulary → two-move completion");

  // Exercise 3: dependent variance jams, then Welford exposes μ and M2.
  await chooseLevel(/Carry the moments/);
  await capture("level3-goal");
  await startLevel();
  assert.equal(await page.getByRole("button", { name: /Lemma/ }).count(), 0);
  await dropPartition("r", "stream");
  assert.match(await page.getByTestId("lemma-wall").innerText(), /whole row/);
  assert.equal(await page.getByRole("button", { name: /Lemma/ }).count(), 1);
  await capture("level3-jam");
  await page.getByRole("button", { name: /Lemma/ }).click();
  await page.locator('[data-box="variance"]').click({ force: true });
  const welfordInspection = await page
    .getByTestId("inspection-variance")
    .innerText();
  assert.match(welfordInspection, /μ/);
  assert.match(welfordInspection, /M2/);
  assert.match(welfordInspection, /δ²/);
  await capture("level3-lemma-inspection");
  await paint("mid1", 2);
  await paint("mid2", 4);
  await dropPartition("r", "stream");
  await capture("level3-completion");
  await expectComplete("Carry the moments", 4);
  console.log(
    "PASS level 3: jam → Welford unlock → inspected carried state → completion",
  );

  // Exercise 8: the larger lemma wall exposes online-softmax reference-frame state.
  await chooseLevel(/Cross the lemma wall/);
  await capture("level8-goal");
  await startLevel();
  assert.equal(await page.getByRole("button", { name: /Lemma/ }).count(), 0);
  await dropPartition("r", "stream");
  assert.match(
    await page.getByTestId("lemma-wall").innerText(),
    /maximum unavailable/,
  );
  assert.equal(await page.getByRole("button", { name: /Lemma/ }).count(), 1);
  await capture("level8-jam");
  await page.getByRole("button", { name: /Lemma/ }).click();
  await page.locator('[data-box="softmax-sum"]').click({ force: true });
  const softmaxInspection = await page
    .getByTestId("inspection-softmax-sum")
    .innerText();
  assert.match(softmaxInspection, /running maximum/);
  assert.match(softmaxInspection, /running normalizer/);
  assert.match(softmaxInspection, /exp\(m_old − m_new\)/);
  await capture("level8-lemma-inspection");
  await paint("mid1", 2);
  await paint("mid2", 4);
  await dropPartition("r", "stream");
  await capture("level8-completion");
  await expectComplete("Cross the lemma wall", 4);
  console.log(
    "PASS level 8: jam → online-softmax unlock → correction inspected → completion",
  );

  // Exercise 9: no autoplay; compose the four learned mechanics by hand.
  await chooseLevel(/Assemble FlashAttention/);
  await capture("level9-goal");
  await startLevel();
  assert.equal(await page.getByRole("button", { name: /Lemma/ }).count(), 0);
  await dropPartition("x", "stream");
  assert.equal(await page.getByRole("button", { name: /Lemma/ }).count(), 1);
  await capture("level9-jam");
  await page.getByRole("button", { name: /Lemma/ }).click();
  await page.locator('[data-box="softmax"]').click({ force: true });
  assert.match(
    await page.getByTestId("inspection-softmax").innerText(),
    /exp\(m_old − m_new\)/,
  );
  await capture("level9-lemma-inspection");
  await paint("scores", 2);
  await paint("probabilities", 4);
  await dropPartition("q", "group");
  await dropPartition("x", "stream");
  await capture("level9-completion");
  await expectComplete("Assemble FlashAttention", 5);
  console.log(
    "PASS level 9: five learned moves → FlashAttention target; ledger recorded",
  );

  assert.deepEqual(errors, []);
  console.log("NCD game-loop acceptance: 4/4 level paths passed");
} finally {
  await browser?.close();
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    // Preview may already have exited after a failed assertion.
  }
}
