/**
 * Task #71 cross-offset replay gate (the offset-0-builds-first trap, now a
 * permanent regression gate).
 *
 * A compiled plan is built ONCE from whichever instance builds the template
 * first. If a view's element offset were baked into the template (or frozen
 * into the recorded params), a start=A build would produce a kernel that reads
 * A's region for EVERY sibling replay — a start=B replay would silently read
 * A's data with B's shape (correct-looking, wrong values). The falsified
 * shortcut passed the TEMPLATE-COUNT check but failed exactly this.
 *
 * This probe: warm + compile a narrow-consume template at offset A (run it
 * enough times that the compiled plan builds and replays), THEN run the SAME
 * template at a DIFFERENT offset B and assert B's values against an independent
 * JS reference. It also interleaves offsets to catch a plan that binds the
 * first-seen offset. Exits 0 on all-correct, 1 on any mismatch.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-view-offset-cross-replay.ts
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const BASE = 64;
  const LEN = 4;
  const baseArr = new Float32Array(BASE * LEN);
  for (let i = 0; i < baseArr.length; i++) baseArr[i] = i;
  const base = api.tensorFromArray(baseArr, [BASE, LEN]);
  const addend = api.tensorFromArray(new Float32Array(LEN * LEN).fill(1), [
    LEN,
    LEN,
  ]);

  // The consume: contiguous(narrow) + add — same shape as the probe, so the
  // template is offset-independent (offset is data).
  const run = async (start: number): Promise<Float32Array> => {
    const view = base.narrow(0, start, LEN);
    const out = api.add(view.contiguous(), addend);
    return new Float32Array(await out.cpu());
  };

  const ref = (start: number): number[] => {
    const r: number[] = [];
    for (let i = 0; i < LEN; i++)
      for (let c = 0; c < LEN; c++) r.push(baseArr[(start + i) * LEN + c] + 1);
    return r;
  };

  const check = (start: number, got: Float32Array): boolean => {
    const exp = ref(start);
    for (let i = 0; i < exp.length; i++) {
      if (got[i] !== exp[i]) {
        console.log(
          `  MISMATCH start=${start} [${i}] got=${got[i]} expect=${exp[i]}`,
        );
        return false;
      }
    }
    return true;
  };

  let allOk = true;

  // Phase 1: build+compile the template at offset A=0 (many reps → 2nd+ exec
  // builds the compiled plan and replays it).
  const A = 0;
  for (let rep = 0; rep < 6; rep++) {
    const got = await run(A);
    if (!check(A, got)) allOk = false;
  }

  // Phase 2: replay the SAME template at DIFFERENT offsets. If offset were
  // frozen at A=0's build, these all read row 0's region (wrong).
  for (const B of [16, 40, 8, 60, 4]) {
    const got = await run(B);
    if (!check(B, got)) {
      allOk = false;
      console.log(`  cross-offset replay FAILED at B=${B} (built at A=${A})`);
    }
  }

  // Phase 3: interleave A and B rapidly (catches a plan that binds the first
  // offset seen within a replay window).
  for (const s of [0, 32, 0, 12, 44, 0, 28]) {
    const got = await run(s);
    if (!check(s, got)) allOk = false;
  }

  console.log(
    allOk
      ? "CROSS-OFFSET-REPLAY: PASS"
      : "CROSS-OFFSET-REPLAY: FAIL",
  );
  process.exit(allOk ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
