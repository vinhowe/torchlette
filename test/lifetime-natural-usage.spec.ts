/**
 * Lifetime contract: many `tensorFromArray + copy_` calls inside one
 * `beginStep`/`markStep` cycle must produce correct results without any
 * user-side reference holding (`refs[]`, `flushStep`-every-N, `void refs`).
 *
 * The framework's job is to keep an `tensorFromArray` source alive long enough
 * for any queued op that reads it to actually consume it. If that contract
 * holds, the agent's F16W upload and outer-optimizer step can drop their
 * scaffolding.
 */

import { beforeAll, describe, expect, it } from "vitest";
import type { DeviceKind } from "../src/backend/types";
import { Torchlette } from "../src/frontend/torchlette";
import { canUseWebGPU } from "./helpers/webgpu";

const TIMEOUT = 120_000;

function expectAllClose(
  actual: number[] | Float32Array,
  expected: number[] | Float32Array,
  atol: number,
  label: string,
): void {
  const a = actual instanceof Float32Array ? actual : Float32Array.from(actual);
  const e =
    expected instanceof Float32Array ? expected : Float32Array.from(expected);
  expect(a.length).toBe(e.length);
  let maxAbs = 0;
  let firstBad = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - e[i]);
    if (d > maxAbs) maxAbs = d;
    if (d > atol && firstBad < 0) firstBad = i;
  }
  if (firstBad >= 0) {
    throw new Error(
      `${label}: max abs diff ${maxAbs} > ${atol} (idx ${firstBad}: ` +
        `actual ${a[firstBad]} vs expected ${e[firstBad]})`,
    );
  }
}

/** Build a distinct Float32Array per param so cross-param aliasing is detectable. */
function makeExpected(shapes: number[][]): Float32Array[] {
  return shapes.map((shape, i) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const arr = new Float32Array(n);
    for (let j = 0; j < n; j++) arr[j] = (i + 1) * 1e-4 + j * 1e-7;
    return arr;
  });
}

async function verifyAll(
  params: ReturnType<Torchlette["zeros"]>[],
  expected: Float32Array[],
  atol: number,
  label: string,
): Promise<void> {
  for (let i = 0; i < params.length; i++) {
    const got = await params[i].cpu();
    expectAllClose(got, expected[i], atol, `${label}[${i}]`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenarios — each takes (api, device) and asserts the contract holds.
// ─────────────────────────────────────────────────────────────────────────────

/** Many uploads inside one beginStep/markStep, no user-side reference holding. */
async function scenarioFlat(api: Torchlette, device: DeviceKind): Promise<void> {
  const shapes: number[][] = [
    ...Array.from({ length: 30 }, () => [768]),
    ...Array.from({ length: 20 }, () => [768, 768]),
    ...Array.from({ length: 20 }, () => [3072, 768]),
    ...Array.from({ length: 10 }, () => [1024, 64]),
  ];
  const params = shapes.map((s) => api.zeros(s, { device, dtype: "f32" }));
  const expected = makeExpected(shapes);

  await api.markStep();
  await api.beginStep();
  for (let i = 0; i < params.length; i++) {
    const t = api.tensorFromArray(new Float32Array(expected[i]), shapes[i], {
      device,
    });
    api.copy_(params[i], t);
    // No refs[]. No flushStep. No void refs.
  }
  api.endStep();
  await api.markStep();

  await verifyAll(params, expected, 1e-6, "flat");
}

/** Same as flat, but force V8 GC mid-loop to stress tensor-finalization paths. */
async function scenarioGCPressure(
  api: Torchlette,
  device: DeviceKind,
): Promise<void> {
  const gc = (globalThis as { gc?: () => void }).gc;
  const shapes: number[][] = [
    ...Array.from({ length: 50 }, () => [512, 64]),
    ...Array.from({ length: 50 }, () => [1024]),
    ...Array.from({ length: 20 }, () => [2048, 256]),
  ];
  const params = shapes.map((s) => api.zeros(s, { device, dtype: "f32" }));
  const expected = makeExpected(shapes);

  await api.markStep();
  await api.beginStep();
  for (let i = 0; i < params.length; i++) {
    const t = api.tensorFromArray(new Float32Array(expected[i]), shapes[i], {
      device,
    });
    api.copy_(params[i], t);
    if (gc && i % 17 === 0) {
      await new Promise((r) => setImmediate(r));
      gc();
    }
  }
  api.endStep();
  await api.markStep();

  await verifyAll(params, expected, 1e-6, "gc");
}

/**
 * Two consecutive upload phases inside one step (DiLoCo: F16W weights, then
 * outer-optimizer). Phase B's writes must overwrite phase A's, not race with
 * phase A's still-pending dispatches.
 */
async function scenarioDiLoCo(
  api: Torchlette,
  device: DeviceKind,
): Promise<void> {
  const numParams = 60;
  const shapes: number[][] = Array.from({ length: numParams }, (_, i) => [
    256,
    64 + (i % 8),
  ]);
  const params = shapes.map((s) => api.zeros(s, { device, dtype: "f32" }));

  const phaseA = makeExpected(shapes);
  const phaseB = phaseA.map((arr) => {
    const out = new Float32Array(arr.length);
    for (let j = 0; j < arr.length; j++) out[j] = arr[j] + 0.5;
    return out;
  });

  await api.markStep();
  await api.beginStep();
  for (let i = 0; i < numParams; i++) {
    api.copy_(
      params[i],
      api.tensorFromArray(phaseA[i], shapes[i], { device }),
    );
  }
  for (let i = 0; i < numParams; i++) {
    api.copy_(
      params[i],
      api.tensorFromArray(phaseB[i], shapes[i], { device }),
    );
  }
  api.endStep();
  await api.markStep();

  await verifyAll(params, phaseB, 1e-5, "diloco");
}

/**
 * The historically-broken pattern: `markStep + dispose + beginStep` *inside*
 * the upload loop. This split the work across multiple plans inside one
 * shared-encoder scope, which is exactly when arena buffer reuse can race
 * with not-yet-submitted dispatches from the prior plan.
 */
async function scenarioMidLoopMarkStep(
  api: Torchlette,
  device: DeviceKind,
): Promise<void> {
  const numParams = 40;
  const shape = [256, 64];
  const params = Array.from({ length: numParams }, () =>
    api.zeros(shape, { device, dtype: "f32" }),
  );
  const expected = makeExpected(Array.from({ length: numParams }, () => shape));

  await api.markStep();
  await api.beginStep();
  let batch: ReturnType<Torchlette["tensorFromArray"]>[] = [];
  for (let i = 0; i < numParams; i++) {
    const t = api.tensorFromArray(expected[i], shape, { device });
    api.copy_(params[i], t);
    batch.push(t);
    if (i % 20 === 19) {
      await api._runtime().forceAllPending();
      await api.markStep();
      for (const tt of batch) tt.dispose();
      batch = [];
      await api.beginStep();
      await new Promise((r) => setImmediate(r));
    }
  }
  api.endStep();
  await api.markStep();

  await verifyAll(params, expected, 1e-6, "midloop");
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite — same contract on CPU and WebGPU.
// ─────────────────────────────────────────────────────────────────────────────

type Scenario = {
  name: string;
  run: (api: Torchlette, device: DeviceKind) => Promise<void>;
};

const SCENARIOS: Scenario[] = [
  { name: "flat upload", run: scenarioFlat },
  { name: "GC pressure mid-loop", run: scenarioGCPressure },
  { name: "DiLoCo two-phase upload", run: scenarioDiLoCo },
  { name: "mid-loop markStep+dispose+beginStep", run: scenarioMidLoopMarkStep },
];

describe("lifetime contract: tensorFromArray + copy_ in one step (CPU)", () => {
  for (const s of SCENARIOS) {
    it(
      s.name,
      async () => {
        const api = new Torchlette("cpu");
        await s.run(api, "cpu");
      },
      TIMEOUT,
    );
  }
});

describe("lifetime contract: tensorFromArray + copy_ in one step (WebGPU)", () => {
  let webgpuAvailable = false;
  beforeAll(async () => {
    webgpuAvailable = await canUseWebGPU();
  });

  for (const s of SCENARIOS) {
    it(
      s.name,
      async () => {
        if (!webgpuAvailable) return;
        const api = new Torchlette("webgpu", { enableFusion: true });
        await s.run(api, "webgpu");
      },
      TIMEOUT,
    );
  }
});
