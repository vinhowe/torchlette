/**
 * WebGPU Buffer Reuse Safety Tests (Minimal)
 *
 * Tests whether Dawn properly orders work between successive queue.submit()
 * calls when buffers are reused. Uses tiny buffers and minimal GPU work
 * to avoid Dawn Node.js threading/hang issues.
 *
 * Usage: npx tsx tools/webgpu-buffer-reuse-test.ts
 */

type WebGPUModule = {
  create: (args: string[]) => { requestAdapter(): Promise<GPUAdapter | null> };
  globals: Record<string, unknown>;
};

let device: GPUDevice;
let queue: GPUQueue;
let copyPL: GPUComputePipeline;

async function initDawn(): Promise<void> {
  const mod = (await import("webgpu")) as unknown as WebGPUModule;
  Object.assign(globalThis, mod.globals);
  const opts: string[] = [];
  if (process.platform === "linux")
    opts.push("enable-dawn-features=vulkan_enable_f16_on_nvidia");
  const provider = mod.create(opts);
  const adapter = await provider.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  device = await adapter.requestDevice();
  queue = device.queue;

  copyPL = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> src: array<f32>;
          @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            if (idx < arrayLength(&src)) { dst[idx] = src[idx]; }
          }`,
      }),
      entryPoint: "main",
    },
  });
}

const N = 1024; // 4KB per buffer — minimal
const BYTES = N * 4;

function sbuf(): GPUBuffer {
  return device.createBuffer({
    size: BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

function fill(b: GPUBuffer, val: number): void {
  queue.writeBuffer(b, 0, new Float32Array(N).fill(val));
}

function computeCopy(enc: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void {
  const bg = device.createBindGroup({
    layout: copyPL.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
    ],
  });
  const pass = enc.beginComputePass();
  pass.setPipeline(copyPL);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(N / 64));
  pass.end();
}

/** Read a buffer back to CPU (creates staging + mapAsync). */
async function readBuf(buf: GPUBuffer): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: BYTES,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, BYTES);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, BYTES);
  const data = new Float32Array(staging.getMappedRange(0, BYTES).slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

function checkAll(data: Float32Array, expected: number): number {
  let bad = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] !== expected) bad++;
  }
  return bad;
}

// ---------------------------------------------------------------------------
// Tests — each is self-contained with ONE readback
// ---------------------------------------------------------------------------

/** submit(B→C), writeBuffer(B,999), C should still be 1 */
async function t1(): Promise<boolean> {
  const B = sbuf(), C = sbuf();
  fill(B, 1);
  const e = device.createCommandEncoder();
  computeCopy(e, B, C);
  queue.submit([e.finish()]);
  fill(B, 999); // overwrite source immediately
  const data = await readBuf(C);
  const bad = checkAll(data, 1);
  console.log(`  1. WAR writeBuffer:            ${bad === 0 ? "PASS" : `FAIL ${bad}/${N}`}`);
  return bad === 0;
}

/** submit(B→C), submit(D→B), C==1, B==2 */
async function t2(): Promise<boolean> {
  const B = sbuf(), C = sbuf(), D = sbuf();
  fill(B, 1); fill(D, 2);
  let e = device.createCommandEncoder();
  computeCopy(e, B, C);
  queue.submit([e.finish()]);
  e = device.createCommandEncoder();
  computeCopy(e, D, B);
  queue.submit([e.finish()]);
  const dataC = await readBuf(C);
  const dataB = await readBuf(B);
  const badC = checkAll(dataC, 1);
  const badB = checkAll(dataB, 2);
  const p = badC === 0 && badB === 0;
  console.log(`  2. WAR compute:                ${p ? "PASS" : `FAIL C:${badC} B:${badB}`}`);
  return p;
}

/** 5 cycles: fill(B,v)→copy→overwrite, verify all outputs */
async function t3(): Promise<boolean> {
  const CYCLES = 5;
  const B = sbuf();
  const outs: GPUBuffer[] = [];
  for (let c = 0; c < CYCLES; c++) {
    fill(B, c + 1);
    outs.push(sbuf());
    const e = device.createCommandEncoder();
    computeCopy(e, B, outs[c]);
    queue.submit([e.finish()]);
    fill(B, -1);
  }
  let fails = 0;
  for (let c = 0; c < CYCLES; c++) {
    const data = await readBuf(outs[c]);
    const bad = checkAll(data, c + 1);
    if (bad > 0) { fails++; console.log(`    cycle ${c}: ${bad}/${N} bad`); }
  }
  console.log(`  3. Rapid chain (${CYCLES} cycles):    ${fails === 0 ? "PASS" : `FAIL ${fails}/${CYCLES}`}`);
  return fails === 0;
}

/** Shared encoder fwd+bwd, submit, reuse intermediates in optimizer */
async function t4(): Promise<boolean> {
  const L = 4;
  const ps: GPUBuffer[] = [], is_: GPUBuffer[] = [],
        os: GPUBuffer[] = [], gs: GPUBuffer[] = [];
  for (let i = 0; i < L; i++) {
    ps.push(sbuf()); is_.push(sbuf()); os.push(sbuf()); gs.push(sbuf());
    fill(ps[i], i + 1);
  }
  const se = device.createCommandEncoder();
  for (let i = 0; i < L; i++) computeCopy(se, ps[i], is_[i]);
  for (let i = 0; i < L; i++) computeCopy(se, is_[i], os[i]);
  for (let i = L - 1; i >= 0; i--) computeCopy(se, os[i], gs[i]);
  queue.submit([se.finish()]);
  const oe = device.createCommandEncoder();
  for (let i = 0; i < L; i++) computeCopy(oe, gs[i], is_[i]);
  queue.submit([oe.finish()]);
  let p = true;
  for (let i = 0; i < L; i++) {
    const dataO = await readBuf(os[i]);
    const dataG = await readBuf(gs[i]);
    const badO = checkAll(dataO, i + 1);
    const badG = checkAll(dataG, i + 1);
    if (badO > 0 || badG > 0) { p = false; console.log(`    L${i} o:${badO} g:${badG}`); }
  }
  console.log(`  4. Shared encoder (${L} layers):  ${p ? "PASS" : "FAIL"}`);
  return p;
}

/** Pool cycling — 3 rounds × 2 bufs */
async function t5(): Promise<boolean> {
  const ROUNDS = 3, PER = 2;
  const pool: GPUBuffer[] = [];
  const dsts: { b: GPUBuffer; v: number }[] = [];
  for (let f = 0; f < ROUNDS; f++) {
    const srcs: GPUBuffer[] = [];
    for (let b = 0; b < PER; b++) {
      srcs.push(pool.pop() ?? sbuf());
      const dst = sbuf();
      fill(srcs[b], f * 100 + b + 1);
      const e = device.createCommandEncoder();
      computeCopy(e, srcs[b], dst);
      queue.submit([e.finish()]);
      dsts.push({ b: dst, v: f * 100 + b + 1 });
    }
    for (const s of srcs) pool.push(s);
  }
  let fails = 0;
  for (const { b, v } of dsts) {
    const data = await readBuf(b);
    if (checkAll(data, v) > 0) fails++;
  }
  console.log(`  5. Multi-flush cycling (${ROUNDS}x${PER}):  ${fails === 0 ? "PASS" : `FAIL ${fails}/${dsts.length}`}`);
  return fails === 0;
}

/** Adam-loop: backward submit, reclaim intermediates, per-param opt submit */
async function t6(): Promise<boolean> {
  const NP = 5;
  const ps: GPUBuffer[] = [], gs: GPUBuffer[] = [], is_: GPUBuffer[] = [];
  for (let i = 0; i < NP; i++) {
    ps.push(sbuf()); gs.push(sbuf()); is_.push(sbuf());
    fill(ps[i], i + 1);
  }
  const be = device.createCommandEncoder();
  for (let i = 0; i < NP; i++) { computeCopy(be, ps[i], is_[i]); computeCopy(be, is_[i], gs[i]); }
  queue.submit([be.finish()]);
  const pool = [...is_];
  for (let i = 0; i < NP; i++) {
    const tmp = pool.pop() ?? sbuf();
    const e = device.createCommandEncoder();
    computeCopy(e, gs[i], tmp);
    computeCopy(e, tmp, ps[i]);
    queue.submit([e.finish()]);
    pool.push(tmp);
  }
  let p = true;
  for (let i = 0; i < NP; i++) {
    const dataG = await readBuf(gs[i]);
    const dataP = await readBuf(ps[i]);
    if (checkAll(dataG, i + 1) > 0 || checkAll(dataP, i + 1) > 0) {
      p = false; console.log(`    P${i} g:${checkAll(dataG, i + 1)} p:${checkAll(dataP, i + 1)}`);
    }
  }
  console.log(`  6. Adam-loop (${NP} params):       ${p ? "PASS" : "FAIL"}`);
  return p;
}

/** Intra-encoder A→B→C + cross-submit R+W */
async function t7t8(): Promise<boolean[]> {
  // Test 7
  const A7 = sbuf(), B7 = sbuf(), C7 = sbuf();
  fill(A7, 42);
  let e = device.createCommandEncoder();
  computeCopy(e, A7, B7); computeCopy(e, B7, C7);
  queue.submit([e.finish()]);
  const d7 = await readBuf(C7);
  const p7 = checkAll(d7, 42) === 0;
  console.log(`  7. Intra-encoder A->B->C:      ${p7 ? "PASS" : `FAIL ${checkAll(d7, 42)}/${N}`}`);

  // Test 8
  const A8 = sbuf(), B8 = sbuf(), C8 = sbuf(), D8 = sbuf();
  fill(A8, 7); fill(D8, 99);
  e = device.createCommandEncoder();
  computeCopy(e, A8, B8); queue.submit([e.finish()]);
  e = device.createCommandEncoder();
  computeCopy(e, B8, C8); computeCopy(e, D8, B8);
  queue.submit([e.finish()]);
  const dC = await readBuf(C8);
  const dB = await readBuf(B8);
  const p8 = checkAll(dC, 7) === 0 && checkAll(dB, 99) === 0;
  console.log(`  8. Cross-submit R+W same buf:  ${p8 ? "PASS" : `FAIL C:${checkAll(dC, 7)} B:${checkAll(dB, 99)}`}`);

  return [p7, p8];
}

// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  await initDawn();
  console.log("=== WebGPU Buffer Reuse Safety Tests ===");
  console.log(`    (${N} f32 elements = ${BYTES / 1024}KB per buffer)\n`);
  const r: boolean[] = [];

  console.log("--- Cross-submission WAR ---");
  r.push(await t1());
  r.push(await t2());

  console.log("\n--- Rapid cycling ---");
  r.push(await t3());

  console.log("\n--- Framework patterns ---");
  r.push(await t4());
  r.push(await t5());
  r.push(await t6());

  console.log("\n--- Intra-encoder ---");
  r.push(...await t7t8());

  const passed = r.filter(Boolean).length;
  console.log(`\n${"=".repeat(48)}`);
  console.log(`RESULTS: ${passed}/${r.length} tests passed`);
  console.log(`${"=".repeat(48)}`);
  if (passed === r.length) {
    console.log("\nAll passed. Dawn properly orders work between");
    console.log("submissions. Buffer reuse across queue.submit()");
    console.log("is safe — framework corruption is a CPU-side");
    console.log("lifetime/bookkeeping error, not a GPU sync issue.");
  } else {
    console.log("\nFAILURES — Dawn may violate submission ordering.");
  }
  device.destroy();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
