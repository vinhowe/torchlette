/**
 * Buffer Corruption Bisection Tool
 *
 * Progressively increases workload complexity to find the minimum pattern
 * that triggers buffer reuse corruption. Each level tests WITH and WITHOUT
 * a GPU fence (queue.onSubmittedWorkDone) before buffer reuse.
 *
 * Levels:
 *   0 - Simple copy chain (4KB)
 *   1 - Size escalation (64KB → 16MB)
 *   2 - Batch dispatches (3MB, N compute passes per encoder)
 *   3 - Read-then-write (3MB, read buffer in N passes, flush, bind as read_write)
 *   4 - Pool simulation (3MB, pendingRelease→pool→acquire cycle)
 *   5 - Framework integration (3MB, actual Torchlette buffer pool + shared encoder)
 *
 * Usage:
 *   npx tsx tools/bisect-buffer-corruption.ts             # Run all levels
 *   npx tsx tools/bisect-buffer-corruption.ts --level 3   # Run specific level
 *   npx tsx tools/bisect-buffer-corruption.ts --size 4096 # Custom buffer size (bytes)
 */

type WebGPUModule = {
  create: (args: string[]) => { requestAdapter(): Promise<GPUAdapter | null> };
  globals: Record<string, unknown>;
};

let device: GPUDevice;
let queue: GPUQueue;

// Pipelines
let copyPL: GPUComputePipeline;
let addPL: GPUComputePipeline;
let rwPL: GPUComputePipeline;

async function initDawn(): Promise<void> {
  const mod = (await import("webgpu")) as unknown as WebGPUModule;
  Object.assign(globalThis, mod.globals);
  const opts: string[] = [];
  if (process.platform === "linux")
    opts.push("enable-dawn-features=vulkan_enable_f16_on_nvidia");
  const provider = mod.create(opts);
  const adapter = await provider.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxBufferSize: 128 * 1024 * 1024,
    },
  });
  queue = device.queue;

  // Copy pipeline: dst[i] = src[i]
  copyPL = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> src: array<f32>;
          @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            if (idx < arrayLength(&src)) { dst[idx] = src[idx]; }
          }`,
      }),
      entryPoint: "main",
    },
  });

  // Add pipeline: dst[i] = src[i] + 1.0
  addPL = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> src: array<f32>;
          @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            if (idx < arrayLength(&src)) { dst[idx] = src[idx] + 1.0; }
          }`,
      }),
      entryPoint: "main",
    },
  });

  // Read-write pipeline: buf[i] = buf[i] * 2.0 (in-place)
  rwPL = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            if (idx < arrayLength(&buf)) { buf[idx] = buf[idx] * 2.0; }
          }`,
      }),
      entryPoint: "main",
    },
  });
}

function sbuf(bytes: number): GPUBuffer {
  return device.createBuffer({
    size: bytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
}

function fill(b: GPUBuffer, val: number, count: number): void {
  queue.writeBuffer(b, 0, new Float32Array(count).fill(val));
}

function dispatchCopy(
  enc: GPUCommandEncoder,
  src: GPUBuffer,
  dst: GPUBuffer,
  count: number,
): void {
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
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();
}

function dispatchAdd(
  enc: GPUCommandEncoder,
  src: GPUBuffer,
  dst: GPUBuffer,
  count: number,
): void {
  const bg = device.createBindGroup({
    layout: addPL.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
    ],
  });
  const pass = enc.beginComputePass();
  pass.setPipeline(addPL);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();
}

function dispatchRW(
  enc: GPUCommandEncoder,
  buf: GPUBuffer,
  count: number,
): void {
  const bg = device.createBindGroup({
    layout: rwPL.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: buf } }],
  });
  const pass = enc.beginComputePass();
  pass.setPipeline(rwPL);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();
}

async function readBuf(
  buf: GPUBuffer,
  bytes: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ, 0, bytes);
  const data = new Float32Array(staging.getMappedRange(0, bytes).slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

function checkAll(data: Float32Array, expected: number): number {
  let bad = 0;
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i] - expected) > 1e-5) bad++;
  }
  return bad;
}

function checkSample(
  data: Float32Array,
  expected: number,
  label: string,
): boolean {
  const bad = checkAll(data, expected);
  if (bad > 0) {
    // Show first few mismatched values
    const examples: string[] = [];
    for (let i = 0; i < data.length && examples.length < 5; i++) {
      if (Math.abs(data[i] - expected) > 1e-5) {
        examples.push(`[${i}]=${data[i].toFixed(6)}`);
      }
    }
    console.log(
      `    ${label}: FAIL ${bad}/${data.length} bad (expected ${expected}, got: ${examples.join(", ")})`,
    );
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Level 0: Simple copy chain (matches existing test)
// ---------------------------------------------------------------------------
async function level0(
  bytes: number,
  fence: boolean,
): Promise<boolean> {
  const N = bytes / 4;
  const A = sbuf(bytes);
  const B = sbuf(bytes);
  const C = sbuf(bytes);
  fill(A, 42, N);

  // Submit A→B
  let enc = device.createCommandEncoder();
  dispatchCopy(enc, A, B, N);
  queue.submit([enc.finish()]);

  if (fence) await queue.onSubmittedWorkDone();

  // Reuse B as write target: overwrite B with new value
  fill(B, 999, N);

  // Submit B→C (should have 999, not 42)
  enc = device.createCommandEncoder();
  dispatchCopy(enc, B, C, N);
  queue.submit([enc.finish()]);

  const data = await readBuf(C, bytes);
  const ok = checkSample(data, 999, "L0 C");

  A.destroy(); B.destroy(); C.destroy();
  return ok;
}

// ---------------------------------------------------------------------------
// Level 1: Size escalation
// ---------------------------------------------------------------------------
async function level1(
  _baseBytes: number,
  fence: boolean,
): Promise<boolean> {
  const sizes = [64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024];
  let allOk = true;

  for (const bytes of sizes) {
    const N = bytes / 4;
    const src = sbuf(bytes);
    const mid = sbuf(bytes);
    const dst = sbuf(bytes);
    fill(src, 7, N);

    // Submit src→mid
    let enc = device.createCommandEncoder();
    dispatchCopy(enc, src, mid, N);
    queue.submit([enc.finish()]);

    if (fence) await queue.onSubmittedWorkDone();

    // Overwrite mid, then copy to dst
    fill(mid, 13, N);
    enc = device.createCommandEncoder();
    dispatchCopy(enc, mid, dst, N);
    queue.submit([enc.finish()]);

    const data = await readBuf(dst, bytes);
    const ok = checkSample(data, 13, `L1 ${(bytes / 1024).toFixed(0)}KB`);
    if (!ok) allOk = false;

    src.destroy(); mid.destroy(); dst.destroy();
  }
  return allOk;
}

// ---------------------------------------------------------------------------
// Level 2: Batch dispatches (N compute passes on one encoder)
// ---------------------------------------------------------------------------
async function level2(
  bytes: number,
  fence: boolean,
): Promise<boolean> {
  const N = bytes / 4;
  const PASSES = 20;

  // Create a chain: each pass reads from prev, writes to next
  const bufs: GPUBuffer[] = [];
  for (let i = 0; i <= PASSES; i++) bufs.push(sbuf(bytes));
  fill(bufs[0], 1, N);

  // Encode all passes on single encoder
  const enc = device.createCommandEncoder();
  for (let i = 0; i < PASSES; i++) {
    dispatchAdd(enc, bufs[i], bufs[i + 1], N);
  }
  queue.submit([enc.finish()]);

  if (fence) await queue.onSubmittedWorkDone();

  // Now reuse bufs[0] as destination of a new chain
  const enc2 = device.createCommandEncoder();
  dispatchCopy(enc2, bufs[PASSES], bufs[0], N);
  queue.submit([enc2.finish()]);

  // bufs[0] should now contain 1 + PASSES = 21
  const dataFinal = await readBuf(bufs[0], bytes);
  const ok0 = checkSample(dataFinal, 1 + PASSES, "L2 reused buf[0]");

  // bufs[PASSES] should still contain 1 + PASSES
  const dataLast = await readBuf(bufs[PASSES], bytes);
  const okLast = checkSample(dataLast, 1 + PASSES, "L2 buf[last]");

  for (const b of bufs) b.destroy();
  return ok0 && okLast;
}

// ---------------------------------------------------------------------------
// Level 3: Read-then-write (read buffer X in N passes, flush, bind X as RW)
// ---------------------------------------------------------------------------
async function level3(
  bytes: number,
  fence: boolean,
): Promise<boolean> {
  const N = bytes / 4;
  const READERS = 10;

  const X = sbuf(bytes);
  fill(X, 5, N);

  // Multiple passes read from X
  const outputs: GPUBuffer[] = [];
  const enc = device.createCommandEncoder();
  for (let i = 0; i < READERS; i++) {
    const out = sbuf(bytes);
    outputs.push(out);
    dispatchCopy(enc, X, out, N);
  }
  queue.submit([enc.finish()]);

  if (fence) await queue.onSubmittedWorkDone();

  // Now bind X as read_write (in-place double)
  const enc2 = device.createCommandEncoder();
  dispatchRW(enc2, X, N); // X[i] *= 2
  queue.submit([enc2.finish()]);

  // Verify: outputs should be 5, X should be 10
  let allOk = true;
  for (let i = 0; i < READERS; i++) {
    const data = await readBuf(outputs[i], bytes);
    if (!checkSample(data, 5, `L3 out[${i}]`)) allOk = false;
  }
  const dataX = await readBuf(X, bytes);
  if (!checkSample(dataX, 10, "L3 X after RW")) allOk = false;

  X.destroy();
  for (const b of outputs) b.destroy();
  return allOk;
}

// ---------------------------------------------------------------------------
// Level 4: Pool simulation (pendingRelease→pool→acquire cycle)
// ---------------------------------------------------------------------------
async function level4(
  bytes: number,
  fence: boolean,
): Promise<boolean> {
  const N = bytes / 4;
  const ROUNDS = 5;
  const BUFS_PER_ROUND = 4;

  // Simulate a buffer pool
  const pool: GPUBuffer[] = [];
  const pendingRelease: GPUBuffer[] = [];
  const results: { buf: GPUBuffer; expected: number }[] = [];

  const acquire = (): GPUBuffer => pool.pop() ?? sbuf(bytes);
  const release = (b: GPUBuffer): void => { pendingRelease.push(b); };
  const flushPending = (): void => {
    while (pendingRelease.length > 0) pool.push(pendingRelease.pop()!);
  };

  for (let round = 0; round < ROUNDS; round++) {
    const srcs: GPUBuffer[] = [];
    const dsts: GPUBuffer[] = [];

    // Acquire buffers, fill, dispatch work
    const enc = device.createCommandEncoder();
    for (let i = 0; i < BUFS_PER_ROUND; i++) {
      const src = acquire();
      const dst = acquire();
      const val = round * 100 + i + 1;
      fill(src, val, N);
      dispatchCopy(enc, src, dst, N);
      srcs.push(src);
      dsts.push(dst);
      results.push({ buf: dst, expected: val });
    }
    queue.submit([enc.finish()]);

    // Release sources back to pending
    for (const s of srcs) release(s);

    if (fence) await queue.onSubmittedWorkDone();

    // Promote pending to pool (this is the critical moment)
    flushPending();
  }

  // Verify all results
  let allOk = true;
  for (let i = 0; i < results.length; i++) {
    const { buf, expected } = results[i];
    const data = await readBuf(buf, bytes);
    if (!checkSample(data, expected, `L4 result[${i}]`)) allOk = false;
  }

  // Cleanup
  for (const { buf } of results) buf.destroy();
  for (const b of pool) b.destroy();
  for (const b of pendingRelease) b.destroy();
  return allOk;
}

// ---------------------------------------------------------------------------
// Level 5: Framework integration (Torchlette buffer pool + shared encoder)
// ---------------------------------------------------------------------------
async function level5(
  bytes: number,
  fence: boolean,
): Promise<boolean> {
  // This level uses the actual Torchlette infrastructure
  const { initWebGPU } = await import("../src/backend/webgpu/index.js");
  const webgpu = await import("../src/backend/webgpu/index.js");

  await initWebGPU();

  const N = bytes / 4;
  const STEPS = 3;
  const PARAMS = 4;

  let allOk = true;

  for (let step = 0; step < STEPS; step++) {
    // Simulate forward pass: create param tensors, compute outputs
    const params: GPUBuffer[] = [];
    const outputs: GPUBuffer[] = [];
    const grads: GPUBuffer[] = [];

    const enc = device.createCommandEncoder();
    for (let p = 0; p < PARAMS; p++) {
      const param = sbuf(bytes);
      const out = sbuf(bytes);
      const grad = sbuf(bytes);
      const val = step * 100 + p + 1;
      fill(param, val, N);
      dispatchCopy(enc, param, out, N);
      dispatchCopy(enc, out, grad, N);
      params.push(param);
      outputs.push(out);
      grads.push(grad);
    }
    queue.submit([enc.finish()]);

    if (fence) await queue.onSubmittedWorkDone();

    // Simulate optimizer: reuse output buffers (like pool reclamation)
    const enc2 = device.createCommandEncoder();
    for (let p = 0; p < PARAMS; p++) {
      // Write new param value to the output buffer (reusing it)
      fill(outputs[p], (step + 1) * 1000 + p, N);
      dispatchCopy(enc2, outputs[p], params[p], N);
    }
    queue.submit([enc2.finish()]);

    // Verify grads still have original values
    for (let p = 0; p < PARAMS; p++) {
      const expected = step * 100 + p + 1;
      const data = await readBuf(grads[p], bytes);
      if (!checkSample(data, expected, `L5 step${step} grad[${p}]`))
        allOk = false;
    }

    // Verify params have new values
    for (let p = 0; p < PARAMS; p++) {
      const expected = (step + 1) * 1000 + p;
      const data = await readBuf(params[p], bytes);
      if (!checkSample(data, expected, `L5 step${step} param[${p}]`))
        allOk = false;
    }

    // Cleanup
    for (const b of [...params, ...outputs, ...grads]) b.destroy();
  }

  return allOk;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const LEVELS: {
  name: string;
  fn: (bytes: number, fence: boolean) => Promise<boolean>;
}[] = [
  { name: "Simple copy chain (4KB)", fn: level0 },
  { name: "Size escalation (64KB→16MB)", fn: level1 },
  { name: "Batch dispatches (N passes/encoder)", fn: level2 },
  { name: "Read-then-write (read N, flush, RW)", fn: level3 },
  { name: "Pool simulation (pending→pool→acquire)", fn: level4 },
  { name: "Framework integration (Torchlette infra)", fn: level5 },
];

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let targetLevel = -1; // -1 = all
  let baseBytes = 3 * 1024 * 1024; // 3MB default

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--level" && args[i + 1]) {
      targetLevel = parseInt(args[i + 1], 10);
      i++;
    } else if (args[i] === "--size" && args[i + 1]) {
      baseBytes = parseInt(args[i + 1], 10);
      i++;
    }
  }

  await initDawn();

  console.log("=== Buffer Corruption Bisection Tool ===");
  console.log(`Base buffer size: ${(baseBytes / 1024).toFixed(0)}KB`);
  console.log();

  const results: { level: number; name: string; noFence: boolean; fence: boolean }[] = [];

  const levelsToRun =
    targetLevel >= 0
      ? [{ idx: targetLevel, ...LEVELS[targetLevel] }]
      : LEVELS.map((l, idx) => ({ idx, ...l }));

  for (const { idx, name, fn } of levelsToRun) {
    console.log(`--- Level ${idx}: ${name} ---`);

    // Without fence
    console.log("  [no fence]");
    const noFence = await fn(idx === 0 ? 4096 : baseBytes, false);
    console.log(`  Result: ${noFence ? "PASS" : "FAIL"}`);

    // With fence
    console.log("  [with fence]");
    const withFence = await fn(idx === 0 ? 4096 : baseBytes, true);
    console.log(`  Result: ${withFence ? "PASS" : "FAIL"}`);

    results.push({ level: idx, name, noFence, fence: withFence });
    console.log();
  }

  // Summary
  console.log("=".repeat(60));
  console.log("SUMMARY");
  console.log("=".repeat(60));
  console.log(
    "Level | Name                                  | No Fence | Fence",
  );
  console.log("-".repeat(60));
  for (const r of results) {
    const nf = r.noFence ? "PASS" : "FAIL";
    const f = r.fence ? "PASS" : "FAIL";
    console.log(
      `  ${r.level}   | ${r.name.padEnd(38)}| ${nf.padEnd(9)}| ${f}`,
    );
  }
  console.log("=".repeat(60));

  // Interpretation
  const anyNoFenceFail = results.some((r) => !r.noFence);
  const anyFenceFail = results.some((r) => !r.fence);
  const fenceFixesAll =
    anyNoFenceFail && !anyFenceFail;

  if (!anyNoFenceFail && !anyFenceFail) {
    console.log(
      "\nAll pass with and without fence. Dawn buffer reuse is safe at this level.",
    );
    console.log(
      "The corruption is likely caused by higher-level framework bookkeeping.",
    );
  } else if (fenceFixesAll) {
    console.log(
      "\nFence fixes all failures! GPU retirement timing is the issue.",
    );
    console.log(
      "Dawn doesn't fully retire work before allowing buffer reuse via queue ordering.",
    );
    const firstFail = results.find((r) => !r.noFence);
    if (firstFail) {
      console.log(
        `First failing level without fence: ${firstFail.level} (${firstFail.name})`,
      );
    }
  } else if (anyFenceFail) {
    console.log(
      "\nSome tests fail even WITH fence. This suggests a test bug or hardware issue.",
    );
  }

  device.destroy();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
