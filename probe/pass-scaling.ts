/**
 * PASS-SCALING benchmark for the step-function-compiler P0 campaign.
 *
 * Times the compile-path passes on the DEFERRED-LOSS whole-step (fwd+bwd) graph
 * and on synthetically-scaled variants (layer replication + honest op mix). The
 * graph is captured at the single backward force (no mid-step loss item), which
 * is exactly the census's DEFERRED-LOSS configuration (630 nodes @ distil/6L).
 *
 * Node count scales ~linearly with numLayers and is INDEPENDENT of hidden size,
 * so we use tiny dims (embed 32, vocab 256) to keep param materialization cheap
 * while honestly reproducing the per-layer op mix (attention/layernorm/gelu/
 * matmul/…). One layer-count per process; the driver invokes it per size.
 *
 * Reproduce a single size:
 *   VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim \
 *     npx tsx probe/pass-scaling.ts <numLayers>
 */
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { RuntimeEngine } from "../src/runtime/engine";
import { getLivePendingNodeIds, getAllPendingTensors } from "../src/runtime/tensor";
import { Adam } from "../src/optim";
import type { LazyIRNode } from "../src/graph/types";
type Tensor = any;

import {
  buildMergedPlan,
  enforceWriteAfterReadOrder,
  tagPlanOutputs,
} from "../src/executor/plan-builder";
import { analyzeGraph } from "../src/compiler/graph-compiler";
import {
  reorderPlanForFusion,
  segmentPlanForExecution,
  detectFusionGroups,
} from "../src/compiler/fusion-detect";
import { runPasses, SIMPLIFICATION_PASSES } from "../src/compiler/graph-rewrites";

// ---- capture harness ----
interface Capture {
  roots: LazyIRNode[];
  live: Set<number>;
}
let captured: Capture | null = null;
const SENTINEL = "__PASS_SCALING_CAPTURED__";

// PS_MODE: "fwdbwd" (default) captures the fwd+bwd graph at the backward force;
// "opt" lets backward execute and captures the optimizer graph (in-place
// adamStep/scatters) at the markStep force — the only graph that exercises the
// WAR pass's splice ready-set / affinity scan / checkpoint all-pairs spots.
const PS_MODE = process.env.PS_MODE ?? "fwdbwd";

function rootsFromTensors(tensors: Tensor[]): LazyIRNode[] {
  const roots: LazyIRNode[] = [];
  for (const t of tensors) {
    if ((t as any).isMaterialized?.() || (t as any).disposed) continue;
    const ref = (t as any).lazyRef;
    if (ref && ref.kind === "pending") roots.push(ref.node);
  }
  return roots;
}

const proto = RuntimeEngine.prototype as any;
const origMerged = proto.forceAllMerged;
const origPending = proto.forceAllPending;
proto.forceAllMerged = async function (...tensors: Tensor[]) {
  if (PS_MODE === "fwdbwd" && !captured) {
    const roots = rootsFromTensors(tensors);
    if (roots.length > 0) {
      captured = { roots, live: new Set(getLivePendingNodeIds()) };
      throw new Error(SENTINEL);
    }
  }
  return origMerged.apply(this, tensors);
};
proto.forceAllPending = async function () {
  if (PS_MODE === "opt" && CAPTURE_ARMED && !captured) {
    // Grab the optimizer roots (all live pending tensors) before executing.
    const roots = rootsFromTensors(getAllPendingTensors());
    if (roots.length > 0) {
      captured = { roots, live: new Set(getLivePendingNodeIds()) };
      throw new Error(SENTINEL);
    }
  }
  return origPending.apply(this, arguments as any);
};
let CAPTURE_ARMED = false;

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Time fn over `iters` runs after `warm` warmups; return median ms. */
function bench(fn: () => void, iters = 9, warm = 3): number {
  for (let i = 0; i < warm; i++) fn();
  const ts: number[] = [];
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    fn();
    ts.push(performance.now() - t0);
  }
  return median(ts);
}

/** Derive externalNodeIds the way the executor does (tagPlanOutputs → outputIndices). */
function externalIds(nodes: LazyIRNode[], live: Set<number>): Set<number> {
  const plan = { nodes } as { nodes: LazyIRNode[]; outputIndices?: number[] };
  tagPlanOutputs(plan as any, live);
  const ids = new Set<number>();
  for (const idx of plan.outputIndices ?? []) ids.add(nodes[idx].id);
  return ids;
}

/**
 * Synthetic WAR-pass stress: a graph of `M` nodes that is a long chain of
 * elementwise ops with a fraction of in-place adamStep nodes (so the fast-path
 * exit is not taken) and a checkpoint boundary every `bstride` nodes (so the
 * all-pairs boundary-edge loop runs). Isolates the checkpoint all-pairs O(B·n)
 * and the Kahn ready-set/affinity spots on the merged whole-step shape without
 * needing a real merged capture.
 */
function synthWarStress(M: number, bstride: number, wide = false): void {
  const nodes: LazyIRNode[] = [];
  const mk = (op: string, inputNodes: LazyIRNode[]): LazyIRNode => {
    const n = {
      id: nodes.length,
      op,
      shape: [8],
      dtype: "f32",
      inputs: inputNodes.map((nn) => ({ kind: "pending", node: nn })),
      isCheckpointBoundary:
        bstride > 0 && nodes.length > 0 && nodes.length % bstride === 0,
    } as unknown as LazyIRNode;
    nodes.push(n);
    return n;
  };
  const root = mk("add", []);
  // wide: a fan-out of `add` nodes all reading the root become ready at once
  // (the backward/optimizer frontier shape); chain: a sequential dependency
  // chain. Every ~5th node is an in-place adamStep whose m/v dsts are DEDICATED
  // leaves (one reader each) — the realistic optimizer shape: WAR edges stay
  // O(n), only the READY-SET width is stressed.
  let prev = root;
  while (nodes.length < M) {
    const base = wide ? root : prev;
    if (nodes.length % 5 === 0) {
      // adamStep inputs = [grad, param, m, v]; dsts 1..3 = param/m/v, each a
      // DEDICATED leaf with a single reader (realistic: WAR edges stay O(n)).
      const p = mk("zeros", []);
      const m = mk("zeros", []);
      const v = mk("zeros", []);
      prev = mk("adamStep", [base, p, m, v]);
    } else {
      prev = mk("add", [base]);
    }
  }
  const t = bench(() => {
    enforceWriteAfterReadOrder(nodes);
  }, 5, 1);
  const nB = nodes.filter((n) => n.isCheckpointBoundary).length;
  console.log(
    "PASS_SCALING_WARSYNTH " +
      JSON.stringify({ M, wide, boundaries: nB, bstride, war: +t.toFixed(3) }),
  );
}

async function main() {
  const numLayers = Number(process.argv[2] ?? 6);
  if (PS_MODE === "warsynth") {
    // No GPU needed — pure node-array stress.
    const M = numLayers; // reuse arg as node count
    synthWarStress(M, 0); // chain, no boundaries
    synthWarStress(M, 100); // chain, boundary every 100 (all-pairs edges)
    synthWarStress(M, 0, true); // WIDE frontier (ready-set/affinity worst case)
    synthWarStress(M, 100, true); // wide + boundaries
    process.exit(0);
  }
  await initWebGPU();

  const config: GPT2Config = {
    vocabSize: 256,
    blockSize: 64,
    numLayers,
    numHeads: 2,
    embedDim: 32,
    dropoutRate: 0,
  };

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = new GPT2(api, config, { device: "webgpu" });
  model.train();
  // From-scratch init leaves params pending; force them so the whole-step
  // graph has materialized param leaves (matches real training post-beginStep).
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);

  const seq = 8;
  const toks = Array.from({ length: seq }, (_, i) => (i * 7 + 3) % config.vocabSize);
  const input = api.tensorFromArray(toks.slice(0, -1), [1, seq - 1]);
  const target = api.tensorFromArray(toks.slice(1), [1, seq - 1]);

  const useCkpt = process.env.PS_CKPT === "1";
  await api.beginStep(); // materialize params
  optimizer.zeroGrad();

  const { loss } = model.forwardWithLoss(input, target, { useCheckpoint: useCkpt });
  try {
    if (PS_MODE === "opt") {
      await loss!.backward(); // executes fwd+bwd
      optimizer.step(); // builds in-place adamStep nodes (lazy)
      api.endStep();
      CAPTURE_ARMED = true;
      await api.markStep(); // → forceAllPending → captures → throws SENTINEL
    } else {
      await loss!.backward(); // intercepted at forceAllMerged → throws SENTINEL
    }
  } catch (e) {
    if (!(e instanceof Error) || e.message !== SENTINEL) throw e;
  }

  if (!captured) {
    console.error(`[pass-scaling] L=${numLayers}: no graph captured`);
    process.exit(1);
  }

  // ---- build the whole-step node array once ----
  const roots = captured.roots;
  const skipExec = PS_MODE === "opt";
  const plan = buildMergedPlan(roots, skipExec);
  const nodes = plan.nodes;
  const N = nodes.length;
  const ext = externalIds(nodes, captured.live);

  // ---- time each pass in isolation ----
  // plan-build: collector (buildMergedPlan) — includes WAR ordering.
  const tBuild = bench(() => {
    buildMergedPlan(roots, skipExec);
  });
  // fusion-reorder: the Kahn + selectBestForFusion scan.
  const tReorder = bench(() => {
    reorderPlanForFusion(nodes);
  });
  // WAR ordering (Kahn ready set + checkpoint edges) in isolation.
  const tWar = bench(() => {
    enforceWriteAfterReadOrder(nodes);
  });
  // CSE + DCE + the other rewrites (redirectConsumers filter + DCE fixpoint).
  const tRewrites = bench(() => {
    const consumers = new Map<number, LazyIRNode[]>();
    const consumerCount = new Map<number, number>();
    for (const node of nodes) {
      for (const inp of node.inputs) {
        if (inp.kind === "pending") {
          consumerCount.set(inp.node.id, (consumerCount.get(inp.node.id) ?? 0) + 1);
          if (!consumers.has(inp.node.id)) consumers.set(inp.node.id, []);
          consumers.get(inp.node.id)!.push(node);
        }
      }
    }
    runPasses({ planNodes: nodes, consumers, consumerCount }, new Set(), SIMPLIFICATION_PASSES);
  });
  // detectFusionGroups alone (proposeCandidateIslands + component split + batching).
  const tDetect = bench(() => {
    detectFusionGroups(nodes, ext, { enableMultiOutput: true });
  });
  // segmentation (detectFusionGroups + the wrapper gap-node walk).
  const tSegment = bench(() => {
    segmentPlanForExecution(nodes, ext, { enableMultiOutput: true });
  });
  // end-to-end compile analysis (everything analyzeGraph does).
  const tAnalyze = bench(() => {
    analyzeGraph(nodes, ext);
  }, 7, 2);

  const nInPlace = nodes.filter((n) => n.op === "adamStep" || n.op === "stridedScatterCopy" || n.op === "stridedScatterAdd").length;
  const nBoundary = nodes.filter((n) => n.isCheckpointBoundary).length;
  const row = {
    mode: PS_MODE + (useCkpt ? "+ckpt" : ""),
    L: numLayers,
    N,
    ext: ext.size,
    inPlace: nInPlace,
    boundaries: nBoundary,
    build: +tBuild.toFixed(3),
    reorder: +tReorder.toFixed(3),
    war: +tWar.toFixed(3),
    rewrites: +tRewrites.toFixed(3),
    detect: +tDetect.toFixed(3),
    segment: +tSegment.toFixed(3),
    analyze: +tAnalyze.toFixed(3),
  };
  console.log("PASS_SCALING_ROW " + JSON.stringify(row));
  process.exit(0);
}
main().catch((e) => {
  console.error(e);
  process.exit(1);
});
