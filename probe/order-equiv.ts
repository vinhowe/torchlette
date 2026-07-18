/**
 * ORDER-EQUIVALENCE probe: my reorderPlanForFusion / enforceWriteAfterReadOrder
 * vs the b8439f67 base versions (inlined below), on the REAL backward and
 * optimizer plans captured from a distilgpt2 training step. Prints the first
 * index where the id-sequences differ. Behavior-identity requires ZERO diffs.
 */
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { RuntimeEngine } from "../src/runtime/engine";
import { getAllPendingTensors } from "../src/runtime/tensor";
import { Adam } from "../src/optim";
import { buildMergedPlan, enforceWriteAfterReadOrder } from "../src/executor/plan-builder";
import { reorderPlanForFusion, isFusibleOp } from "../src/compiler/fusion-detect";
import type { LazyIRNode } from "../src/graph/types";

// ---------- inlined BASE implementations (b8439f67) ----------
const IN_PLACE_DST_INPUTS: Record<string, number[]> = {
  stridedScatterCopy: [0],
  stridedScatterAdd: [0],
  adamStep: [1, 2, 3],
};
function baseHasPendingInputIn(node: LazyIRNode, set: Set<number>): boolean {
  for (const input of node.inputs)
    if (input.kind === "pending" && set.has(input.node.id)) return true;
  return false;
}
function baseReorder(nodes: LazyIRNode[]): LazyIRNode[] {
  if (nodes.length <= 2) return nodes;
  const nodeById = new Map<number, LazyIRNode>();
  const originalPos = new Map<number, number>();
  for (let i = 0; i < nodes.length; i++) {
    nodeById.set(nodes[i].id, nodes[i]);
    originalPos.set(nodes[i].id, i);
  }
  const inDegree = new Map<number, number>();
  const successors = new Map<number, number[]>();
  for (const node of nodes) {
    inDegree.set(node.id, 0);
    successors.set(node.id, []);
  }
  for (const node of nodes)
    for (const input of node.inputs)
      if (input.kind === "pending" && nodeById.has(input.node.id)) {
        inDegree.set(node.id, (inDegree.get(node.id) as number) + 1);
        successors.get(input.node.id)?.push(node.id);
      }
  const ready = new Set<number>();
  for (const node of nodes) if (inDegree.get(node.id) === 0) ready.add(node.id);
  const result: LazyIRNode[] = [];
  let chainNodeIds = new Set<number>();
  const select = (): number => {
    let bestId = -1, bestPriority = 3, bestPos = Infinity;
    for (const id of ready) {
      const node = nodeById.get(id) as LazyIRNode;
      const fus = isFusibleOp(node.op);
      let priority: number;
      if (fus && chainNodeIds.size > 0 && baseHasPendingInputIn(node, chainNodeIds)) priority = 0;
      else if (fus) priority = 1;
      else priority = 2;
      const pos = originalPos.get(id) as number;
      if (priority < bestPriority || (priority === bestPriority && pos < bestPos)) {
        bestId = id; bestPriority = priority; bestPos = pos;
      }
    }
    return bestId;
  };
  while (ready.size > 0) {
    const best = select();
    ready.delete(best);
    const bestNode = nodeById.get(best) as LazyIRNode;
    result.push(bestNode);
    if (isFusibleOp(bestNode.op)) chainNodeIds.add(best);
    else chainNodeIds = new Set();
    for (const succId of successors.get(best) as number[]) {
      const nd = (inDegree.get(succId) as number) - 1;
      inDegree.set(succId, nd);
      if (nd === 0) ready.add(succId);
    }
  }
  return result;
}
function baseEnforce(nodes: LazyIRNode[]): LazyIRNode[] {
  let hasInPlace = false;
  for (const n of nodes) if (IN_PLACE_DST_INPUTS[n.op]) { hasInPlace = true; break; }
  if (!hasInPlace) return nodes;
  const indexOf = new Map<LazyIRNode, number>();
  for (let i = 0; i < nodes.length; i++) indexOf.set(nodes[i], i);
  const readers = new Map<LazyIRNode, Map<number, number[]>>();
  for (let i = 0; i < nodes.length; i++)
    for (const ref of nodes[i].inputs) {
      if (ref.kind !== "pending") continue;
      let byOi = readers.get(ref.node);
      if (!byOi) readers.set(ref.node, (byOi = new Map()));
      const oi = ref.outputIndex ?? 0;
      let list = byOi.get(oi);
      if (!list) byOi.set(oi, (list = []));
      list.push(i);
    }
  const succ: number[][] = nodes.map(() => []);
  const indeg = new Array<number>(nodes.length).fill(0);
  const addEdge = (from: number, to: number) => { succ[from].push(to); indeg[to]++; };
  for (let i = 0; i < nodes.length; i++)
    for (const ref of nodes[i].inputs) {
      if (ref.kind !== "pending") continue;
      const pi = indexOf.get(ref.node);
      if (pi !== undefined) addEdge(pi, i);
    }
  for (let b = 0; b < nodes.length; b++) {
    if (!nodes[b].isCheckpointBoundary) continue;
    for (let i = 0; i < b; i++) addEdge(i, b);
    for (let j = b + 1; j < nodes.length; j++) addEdge(b, j);
  }
  for (let i = 0; i < nodes.length; i++) {
    const dstInputs = IN_PLACE_DST_INPUTS[nodes[i].op];
    if (!dstInputs) continue;
    for (const di of dstInputs) {
      const ref = nodes[i].inputs[di];
      if (!ref || ref.kind !== "pending") continue;
      const list = readers.get(ref.node)?.get(ref.outputIndex ?? 0);
      if (!list) continue;
      for (const ri of list) if (ri !== i) addEdge(ri, i);
    }
  }
  const heap: number[] = [];
  const pushReady = (i: number) => {
    let lo = 0, hi = heap.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (heap[mid] < i) lo = mid + 1; else hi = mid; }
    heap.splice(lo, 0, i);
  };
  for (let i = 0; i < nodes.length; i++) if (indeg[i] === 0) pushReady(i);
  const order: LazyIRNode[] = [];
  let lastOp: string | null = null, lastOpAffine = false;
  while (heap.length > 0) {
    let pick = 0;
    if (lastOpAffine && heap.length > 1)
      for (let h = 0; h < heap.length; h++) if (nodes[heap[h]].op === lastOp) { pick = h; break; }
    const i = heap.splice(pick, 1)[0];
    order.push(nodes[i]);
    lastOp = nodes[i].op;
    lastOpAffine = !isFusibleOp(lastOp);
    for (const j of succ[i]) if (--indeg[j] === 0) pushReady(j);
  }
  if (order.length !== nodes.length) return nodes;
  return order;
}

// ---------- capture ----------
let capBwd: LazyIRNode[] | null = null;
let capOpt: LazyIRNode[] | null = null;
let armedOpt = false;
const proto = RuntimeEngine.prototype as any;
const oM = proto.forceAllMerged, oP = proto.forceAllPending;
function roots(ts: any[]): LazyIRNode[] {
  const r: LazyIRNode[] = [];
  for (const t of ts) { if (t.isMaterialized?.() || t.disposed) continue; const rf = t.lazyRef; if (rf?.kind === "pending") r.push(rf.node); }
  return r;
}
proto.forceAllMerged = async function (...ts: any[]) {
  if (!capBwd) { const r = roots(ts); if (r.length > 1) capBwd = buildMergedPlan(r).nodes; }
  return oM.apply(this, ts);
};
proto.forceAllPending = async function () {
  if (armedOpt && !capOpt) { const r = roots(getAllPendingTensors()); if (r.length) capOpt = buildMergedPlan(r, true).nodes; }
  return oP.apply(this, arguments as any);
};

function ids(ns: LazyIRNode[]) { return ns.map((n) => n.id).join(","); }
function firstDiff(a: LazyIRNode[], b: LazyIRNode[]): string {
  if (a.length !== b.length) return `LENGTH ${a.length} vs ${b.length}`;
  for (let i = 0; i < a.length; i++)
    if (a[i].id !== b[i].id)
      return `idx ${i}: mine=${a[i].id}(${a[i].op}) base=${b[i].id}(${b[i].op})`;
  return "IDENTICAL";
}

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true, enableMemoryPlanning: true, enableCheckpointSegmentation: true });
  const model = await loadPretrainedGPT2(api, "./models/distilgpt2", { dropoutRate: 0 }, { device: "webgpu" });
  model.train();
  const opt = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const toks = [2514, 307, 393, 407, 284, 307, 11, 326, 318, 262, 1808, 13, 2514, 307, 393, 407];
  const input = api.tensorFromArray(toks.slice(0, -1), [1, toks.length - 1]);
  const target = api.tensorFromArray(toks.slice(1), [1, toks.length - 1]);
  await api.beginStep();
  opt.zeroGrad();
  const { loss } = model.forwardWithLoss(input, target, { useCheckpoint: false });
  await loss!.backward();
  opt.step();
  api.endStep();
  armedOpt = true;
  await api.markStep();

  for (const [name, plan] of [["backward", capBwd], ["optimizer", capOpt]] as const) {
    if (!plan) { console.log(`${name}: NOT CAPTURED`); continue; }
    const rMine = reorderPlanForFusion(plan), rBase = baseReorder(plan);
    console.log(`REORDER ${name} (${plan.length} nodes): ${firstDiff(rMine, rBase)}`);
    // enforce runs on reorder output; compare enforce on the SAME (base-reorder) input
    const eMine = enforceWriteAfterReadOrder(rBase), eBase = baseEnforce(rBase);
    console.log(`ENFORCE ${name} (on base-reorder): ${firstDiff(eMine, eBase)}`);
    // and the full composition each side uses
    console.log(`COMPOSE ${name}: ${firstDiff(enforceWriteAfterReadOrder(rMine), baseEnforce(rBase))}`);
  }
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
