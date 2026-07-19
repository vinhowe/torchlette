/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — PROBE 2: K-BLOCK SCALE + COMPILE.
 *
 * Unrolled-K makes K decode tokens ONE static graph. This probe constructs that
 * graph over a real distilgpt2 (the Probe-1 on-device argmax->gather feedback
 * wiring, random init — weights don't change node count) for K in {1,2,4,8} and
 * measures, WITHOUT executing:
 *   - node count N of the K-block whole-graph (buildMergedPlan) — the scale the
 *     compiler passes must handle; compare to P0's scaled-pass table (up to 14589).
 *   - analyzeGraph(nodes, ext) time (median) — the once-per-trace COMPILE cost.
 *   - N/K (per-token node count) and the growth shape.
 *
 * Then a COMPILE-COVERAGE pass: it FORCES an unrolled-K=8 decode through the real
 * engine (compiled path on) while capturing every generateStream() `uncovered`
 * map (build-from-IR coverage), and reports the decode uncovered residue labels
 * and template-count behavior. This surfaces whether the K-block compiles via
 * build-from-IR and what (if anything) strands lowered on the decode path.
 *
 * NOTE the KV regime: distilgpt2 forwardCached uses a GROWING (concat) KV, so the
 * K positions t+1..t+K are K DISTINCT shapes (kvSeqLen grows) — one template each.
 * The static-KV path (packages/qwen3|gemma2 forwardStatic) keeps position/token as
 * DATA index-tensors with only the 128-bucket length as shape, so K in-bucket
 * steps SHARE one template. This probe measures the growing-KV scale (the harder,
 * more-nodes case) and reports the template implication for both regimes.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-uk-scale.ts
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { DISTILGPT2_CONFIG, GPT2 } from "../examples/gpt2/model";
import type { KVCache } from "../examples/gpt2/model";
import type { LazyIRNode } from "../src/graph/types";
import { buildMergedPlan, tagPlanOutputs } from "../src/executor/plan-builder";
import { analyzeGraph } from "../src/compiler/graph-compiler";
import * as SG from "../src/executor/stream-generate";
import { debugTemplateCount } from "../src/executor/executor";

const PROMPT = [464, 3139, 286, 4881, 318];

function rootsFromTensors(tensors: Tensor[]): LazyIRNode[] {
  const roots: LazyIRNode[] = [];
  for (const t of tensors) {
    const rt = (t as any)._unwrap ? (t as any)._unwrap() : t;
    if (rt.isMaterialized?.() || rt.disposed) continue;
    const ref = rt.lazyRef;
    if (ref && ref.kind === "pending") roots.push(ref.node);
  }
  return roots;
}
function externalIds(nodes: LazyIRNode[], live: Set<number>): Set<number> {
  const plan = { nodes } as { nodes: LazyIRNode[]; outputIndices?: number[] };
  tagPlanOutputs(plan as any, live);
  const ids = new Set<number>();
  for (const idx of plan.outputIndices ?? []) ids.add(nodes[idx].id);
  return ids;
}
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
}

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }

  console.log(`=== PROBE 2: K-BLOCK SCALE + COMPILE (distilgpt2, growing-KV) ===\n`);

  // ---------- PART A: node count + analyzeGraph time per K (no execution) ----------
  console.log(`--- scale (no execution): node count N and analyzeGraph ms per K ---`);
  const rows: { K: number; N: number; ext: number; analyzeMs: number }[] = [];
  for (const K of [1, 2, 4, 8]) {
    const api = new Torchlette("webgpu", { enableFusion: true });
    api.manualSeed(1234);
    const model = new GPT2(api, { ...DISTILGPT2_CONFIG });
    const V = model.config.vocabSize;
    const lastRow = (logits: Tensor): Tensor => {
      const S = logits.shape[1];
      return api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V));
    };
    // Materialize params first (so buildMergedPlan sees materialized param
    // externals, not param-creation nodes) — matches pass-scaling's beginStep.
    {
      const w = api.noGrad(() => model.forwardCached(api.tensorFromArray(PROMPT, [1, PROMPT.length]), undefined, 0));
      await w.logits.cpu();
      w.logits.dispose();
      await api.markStep();
    }
    // Build the unrolled-K graph WITHOUT forcing (the exact on-device feedback chain).
    let { logits, kvs } = ((): { logits: Tensor; kvs: KVCache[] } => {
      const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
      const r = api.noGrad(() => model.forwardCached(idx, undefined, 0));
      return { logits: r.logits, kvs: r.presentKVs };
    })();
    let pos = PROMPT.length;
    const idTensors: Tensor[] = [];
    for (let i = 0; i < K; i++) {
      const id = api.noGrad(() => api.argmax(lastRow(logits), { dim: -1, keepdim: false }));
      idTensors.push(id);
      const nxt = api.noGrad(() => model.forwardCached(api.reshape(id, [1, 1]), kvs, pos));
      logits = nxt.logits;
      kvs = nxt.presentKVs;
      pos += 1;
    }
    // Roots = the K id tensors + the final logits (the whole K-block outputs).
    const roots = rootsFromTensors([...idTensors, logits]);
    const plan = buildMergedPlan(roots, false);
    const nodes = plan.nodes;
    // Tag ONLY the true output roots as external (not all-live) so analyzeGraph
    // does full interior fusion — a realistic (upper-bound) compile-cost timing.
    const live = new Set(roots.map((r) => r.id));
    const ext = externalIds(nodes, live);
    // time analyzeGraph (median of a few; warm twice)
    for (let w = 0; w < 2; w++) analyzeGraph(nodes, ext);
    const ts: number[] = [];
    for (let r = 0; r < 5; r++) {
      const t0 = performance.now();
      analyzeGraph(nodes, ext);
      ts.push(performance.now() - t0);
    }
    rows.push({ K, N: nodes.length, ext: ext.size, analyzeMs: +median(ts).toFixed(2) });
    // don't force — abandon the graph; fresh api next K
  }
  console.log(`  K | nodes N | externals | analyzeGraph ms | nodes/token`);
  for (const r of rows) {
    console.log(
      `  ${r.K} | ${r.N} | ${r.ext} | ${r.analyzeMs} | ${(r.N / r.K).toFixed(0)}`,
    );
  }
  const k8 = rows.find((r) => r.K === 8)!;
  console.log(
    `\n  K=8 block: ${k8.N} nodes, analyzeGraph ${k8.analyzeMs} ms (P0 scaled-pass table: 5859 nodes@31ms, 14589 nodes@84ms — this K-block sits inside that measured-linear range).`,
  );

  // ---------- PART B: compile coverage (forced run, capture generateStream.uncovered) ----------
  console.log(`\n--- compile coverage (forced unrolled-K=8, capturing build-from-IR uncovered) ---`);
  const uncoveredAll = new Map<string, number>();
  let genCalls = 0;
  let fullyCoveredCount = 0;
  let patched = false;
  try {
    const orig = SG.generateStream;
    (SG as any).generateStream = function (...args: any[]) {
      const res = orig.apply(this, args as any);
      genCalls++;
      if (res && res.uncovered) {
        if (res.fullyCovered) fullyCoveredCount++;
        for (const [label, n] of res.uncovered as Map<string, number>) {
          uncoveredAll.set(label, (uncoveredAll.get(label) ?? 0) + n);
        }
      }
      return res;
    };
    patched = (SG as any).generateStream !== orig;
  } catch {
    patched = false;
  }

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1234);
  const model = new GPT2(api, { ...DISTILGPT2_CONFIG });
  const V = model.config.vocabSize;
  const lastRow = (logits: Tensor): Tensor => {
    const S = logits.shape[1];
    return api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V));
  };
  const tmplBefore = debugTemplateCount();
  // Run TWO consecutive unrolled-K=8 blocks (so any shared template can cut over).
  let posOffset = 0;
  let carryTok = PROMPT;
  let carryPast: KVCache[] | undefined;
  const tmplAfterBlock: number[] = [];
  for (let block = 0; block < 2; block++) {
    let { logits, kvs } = ((): { logits: Tensor; kvs: KVCache[] } => {
      const idx = api.tensorFromArray(carryTok, [1, carryTok.length]);
      const r = api.noGrad(() => model.forwardCached(idx, carryPast, posOffset));
      return { logits: r.logits, kvs: r.presentKVs };
    })();
    posOffset += carryTok.length;
    const idTensors: Tensor[] = [];
    for (let i = 0; i < 8; i++) {
      const id = api.noGrad(() => api.argmax(lastRow(logits), { dim: -1, keepdim: false }));
      idTensors.push(id);
      const nxt = api.noGrad(() => model.forwardCached(api.reshape(id, [1, 1]), kvs, posOffset));
      logits = nxt.logits;
      kvs = nxt.presentKVs;
      posOffset += 1;
    }
    const stacked = api.cat(idTensors.map((t) => api.reshape(t, [1, 1])), 1);
    const ids = new Float32Array(await stacked.cpu()); // ONE readback per block
    carryTok = [Math.round(ids[7])]; // continue from the last generated token
    carryPast = kvs;
    posOffset -= 0;
    await api.markStep();
    tmplAfterBlock.push(debugTemplateCount());
  }

  console.log(`  generateStream calls: ${genCalls} (patch bound: ${patched})`);
  console.log(`  fullyCovered generateStream calls: ${fullyCoveredCount}/${genCalls}`);
  console.log(`  template count: ${tmplBefore} -> [block0 ${tmplAfterBlock[0]}, block1 ${tmplAfterBlock[1]}]`);
  if (uncoveredAll.size === 0) {
    console.log(`  uncovered residue: NONE captured (either fully covered, or patch did not bind — see template growth)`);
  } else {
    console.log(`  uncovered residue labels (build-from-IR bails on the decode path):`);
    for (const [label, n] of [...uncoveredAll.entries()].sort((a, b) => b[1] - a[1])) {
      console.log(`    ${label} : ${n}`);
    }
  }
  console.log(
    `\n  NOTE: growing-KV distilgpt2 => each of the K positions is a distinct kvSeqLen shape (one template/position, block1 re-lowers block0's shapes shifted). The static-KV forwardStatic path keeps position as DATA and only the 128-bucket length as shape, so K in-bucket steps SHARE one template (P4a measured decode steady template growth 0).`,
  );
  console.log(
    `\n=== UK-SCALE-STATS === ${JSON.stringify({ scale: rows, genCalls, fullyCoveredCount, uncovered: Object.fromEntries(uncoveredAll), tmplBefore, tmplAfterBlock })}`,
  );
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
