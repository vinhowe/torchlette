/**
 * gpu-probe-map.ts — resolve the DYNAMIC Vulkan↔nvidia-smi device mapping.
 *
 * Dawn's VULKAN_DEVICE_INDEX selects the Nth device in *Vulkan* enumeration
 * order (via tools/vk-shim). That order is NOT guaranteed to match nvidia-smi's
 * index order, and it can drift — the collision class that fabricated ~300
 * "failures" was two agents both trusting VULKAN_DEVICE_INDEX==physical-index.
 *
 * This probe resolves the mapping empirically for ONE candidate index by
 * allocate-and-watch: sample nvidia-smi per-device memory.used, spin up Dawn on
 * VULKAN_DEVICE_INDEX, force a modest GPU allocation, re-sample. The physical
 * device whose used-memory jumps (Dawn context + our tensor, typically several
 * hundred MiB) is the one this VULKAN_DEVICE_INDEX actually lands on. The delta
 * isolates OUR footprint, so it is robust to other tenants already resident.
 *
 * Contract (parsed by tools/pick-gpu.sh):
 *   - stdout: exactly one line `PHYS=<idx> DELTA=<mib>`  (PHYS=-1 on failure /
 *     occupied device where the probe alloc could not land).
 *   - everything else goes to stderr.
 *   - always exits 0 after a clean Dawn teardown (Dawn holds background threads;
 *     process.exit is mandatory — see docs/agent-ops.md).
 *
 * Run (pick-gpu.sh does this for you):
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *     npx tsx tools/gpu-probe-map.ts
 *
 * Env: PROBE_MIB (default 256) — MiB of f32 tensor to force resident.
 */
import { execSync } from "node:child_process";

const PROBE_MIB = parseInt(process.env.PROBE_MIB ?? "256", 10);
// A delta this large is unambiguously ours: Dawn's Vulkan context alone is
// usually >200 MiB and we add a forced PROBE_MIB tensor on top.
const DELTA_FLOOR_MIB = Math.max(64, Math.floor(PROBE_MIB * 0.5));
const log = (m: string) => console.error(`[gpu-probe] ${m}`);

function sampleUsedMiB(): Map<number, number> {
  const out = execSync(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
    { encoding: "utf8" },
  );
  const m = new Map<number, number>();
  for (const line of out.trim().split("\n")) {
    const [idx, used] = line.split(",").map((s) => parseInt(s.trim(), 10));
    if (!Number.isNaN(idx)) m.set(idx, used);
  }
  return m;
}

function emit(phys: number, delta: number): void {
  // The ONE machine-parsed line.
  console.log(`PHYS=${phys} DELTA=${delta}`);
}

async function main(): Promise<void> {
  const vkIdx = process.env.VULKAN_DEVICE_INDEX ?? "(unset)";
  log(`probing VULKAN_DEVICE_INDEX=${vkIdx}, alloc=${PROBE_MIB}MiB`);

  const before = sampleUsedMiB();

  // Dynamic imports so a Dawn/init failure is catchable and never aborts the
  // module before we can emit PHYS=-1.
  let destroyWebGPU: (() => void) | undefined;
  try {
    const backend = await import("../src/backend/webgpu");
    destroyWebGPU = backend.destroyWebGPU;
    if (!(await backend.initWebGPU())) {
      log("initWebGPU returned false (no device / occupied)");
      emit(-1, 0);
      process.exit(0);
    }
    const { Torchlette } = await import("../src/frontend/torchlette");
    const api = new Torchlette("webgpu", { enableFusion: false });

    // PROBE_MIB of f32 = PROBE_MIB*2^20/4 elements. Shape it 2-D so it is a
    // plain dense buffer. ones() + sum()+item() forces real residency.
    const elems = Math.floor((PROBE_MIB * 1024 * 1024) / 4);
    const cols = 1024;
    const rows = Math.max(1, Math.floor(elems / cols));
    const t = api.ones([rows, cols], { device: "webgpu" });
    const s = api.sum(api.mul(t, 1.0));
    await api.item(s); // fence — buffer is resident now
  } catch (e: unknown) {
    log(`alloc/init failed: ${(e as Error)?.message ?? e}`);
    try {
      destroyWebGPU?.();
    } catch {
      /* ignore */
    }
    emit(-1, 0);
    process.exit(0);
  }

  const after = sampleUsedMiB();

  let bestPhys = -1;
  let bestDelta = 0;
  for (const [idx, usedAfter] of after) {
    const delta = usedAfter - (before.get(idx) ?? 0);
    if (delta > bestDelta) {
      bestDelta = delta;
      bestPhys = idx;
    }
  }

  try {
    destroyWebGPU?.();
  } catch {
    /* ignore */
  }

  if (bestPhys < 0 || bestDelta < DELTA_FLOOR_MIB) {
    log(
      `no clear device delta (best +${bestDelta}MiB < floor ${DELTA_FLOOR_MIB}MiB) — inconclusive`,
    );
    emit(-1, bestDelta);
  } else {
    log(`VULKAN_DEVICE_INDEX=${vkIdx} -> physical GPU ${bestPhys} (+${bestDelta}MiB)`);
    emit(bestPhys, bestDelta);
  }
  process.exit(0);
}

main();
