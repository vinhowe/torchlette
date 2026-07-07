/**
 * Test whether CUDA_VISIBLE_DEVICES pins Dawn/WebGPU to a specific GPU.
 *
 * Usage:
 *   CUDA_VISIBLE_DEVICES=8 npx tsx tools/gpu-pin-test.ts
 *
 * The test allocates ~512 MB of GPU buffers, then sleeps for 8s so an
 * external observer can run `nvidia-smi --query-compute-apps=pid,used_memory
 * --format=csv` and see which physical GPU index got the allocation.
 */
import { initWebGPU } from "../src/backend/webgpu/gpu-context";
import { requireContext } from "../src/backend/webgpu/webgpu-state";

async function main() {
  const visible = process.env.CUDA_VISIBLE_DEVICES ?? "(unset)";
  console.error(`pid=${process.pid} CUDA_VISIBLE_DEVICES=${visible}`);

  const ok = await initWebGPU();
  if (!ok) {
    console.error("WebGPU init failed");
    process.exit(1);
  }
  const ctx = requireContext();
  const buffers: GPUBuffer[] = [];
  const sizeMB = 64;
  const count = 8;
  for (let i = 0; i < count; i++) {
    const buf = ctx.device.createBuffer({
      size: sizeMB * 1024 * 1024,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    buffers.push(buf);
  }
  // Force the allocations to commit by writing to one of them
  const filler = new Uint8Array(64);
  ctx.device.queue.writeBuffer(buffers[0], 0, filler);
  await ctx.device.queue.onSubmittedWorkDone?.();

  console.error(`allocated ${count * sizeMB} MB across ${count} buffers`);
  console.error(`sleeping 8s — read nvidia-smi now`);
  await new Promise((r) => setTimeout(r, 8000));

  for (const b of buffers) b.destroy();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
