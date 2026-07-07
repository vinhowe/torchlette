/**
 * Diagnose: compare cached vs fresh analysis for the failing case.
 * Test C from bisection: Adam created + warmup + merged backward.
 *
 * Logs the segment structure from the cache-miss analysis (stored on template)
 * vs a fresh analysis of the same plan on cache hit.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";
import { analyzeGraph } from "../src/compiler/graph-compiler";
import { computePlanFingerprint, buildIdPositionMap } from "../src/compiler/fusion-detect";
import { buildMergedPlan } from "../src/executor/plan-builder";
import { getPendingNodeIds } from "../src/runtime/tensor";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const vocabSize = VOCAB_SIZE_DATA * 2 + 1, seqLen = 10, batchSize = 64;

  // ── Torchlette WITHOUT fusion (so we control analysis manually) ──
  const api = new Torchlette("webgpu", { enableFusion: false });
  const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
  const optimizer = new Adam(model.parameters(), { lr: 1e-2 });

  // Warmup: force model weights
  const bw = generateBatchWithCompartments({ seqLen, batchSize: 2 }, 2);
  const tw = api.tensorFromArray(bw.tokens, [2, seqLen], { dtype: "i32" });
  await model.forward(tw).logits.cpu(); tw.dispose();
  console.log("Warmup done.\n");

  // ── Step: build the plans that would be forced during backward ──
  const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
  const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
  const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });

  // Build the forward graph (lazy)
  const loss = api.tidy(() => {
    const fwd = model.forward(tok);
    const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
    const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
  });
  tok.dispose(); tgt.dispose();

  // ── Simulate what backward does: collect pending roots, build plan ──
  // 1. The "saved tensor force" (collectAndForceSavedTensors equivalent)
  const lossRt = loss._unwrap();
  const savedRoots: any[] = [];
  if (!lossRt.isMaterialized()) savedRoots.push(lossRt.lazyRef.node);
  const savedPlan = buildMergedPlan(savedRoots);
  console.log(`Saved-tensor plan: ${savedPlan.nodes.length} nodes`);

  // Get external node IDs (what executePlanOptimized would see)
  const externalNodeIds = getPendingNodeIds();
  console.log(`External node IDs: ${externalNodeIds.size}`);

  // ── Compute fingerprint ──
  const fp = computePlanFingerprint(savedPlan.nodes, externalNodeIds);
  console.log(`Fingerprint: 0x${fp.primary.toString(16)}/${fp.secondary.toString(16)}`);

  // ── Run analysis TWICE with the same inputs ──
  const analysis1 = analyzeGraph(savedPlan.nodes, externalNodeIds, undefined);
  console.log(`\nAnalysis 1 (fresh):`);
  logAnalysis(analysis1, savedPlan);

  // Now force the saved tensors (like collectAndForceSavedTensors does)
  await loss.item(); // forces the forward plan

  // Build a NEW plan for the same structural content (gradient force)
  // After item(), forward nodes are materialized. Backward would build
  // gradient nodes. Let's just re-analyze the same plan structure.
  // Actually, let's build the gradient plan.
  await loss.backward(); loss.dispose();

  // For comparison: rebuild a plan for a SECOND step
  const b2 = generateBatchWithCompartments({ seqLen, batchSize }, 2);
  const tok2 = api.tensorFromArray(b2.tokens, [batchSize, seqLen], { dtype: "i32" });
  const tgt2 = api.tensorFromArray(b2.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
  const loss2 = api.tidy(() => {
    const fwd2 = model.forward(tok2);
    const logits2 = fwd2.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
    const l2 = crossEntropy(api, logits2, tgt2); api.keep(l2); return l2;
  });
  tok2.dispose(); tgt2.dispose();

  // Build plan for step 2's forward
  const loss2Rt = loss2._unwrap();
  const roots2: any[] = [];
  if (!loss2Rt.isMaterialized()) roots2.push(loss2Rt.lazyRef.node);
  const plan2 = buildMergedPlan(roots2);
  console.log(`\nStep 2 plan: ${plan2.nodes.length} nodes`);

  const externalNodeIds2 = getPendingNodeIds();
  const fp2 = computePlanFingerprint(plan2.nodes, externalNodeIds2);
  const fpSame = fp.primary === fp2.primary && fp.secondary === fp2.secondary;
  console.log(`Fingerprint: 0x${fp2.primary.toString(16)}/${fp2.secondary.toString(16)} (same as step 1? ${fpSame})`);
  console.log(`External IDs: ${externalNodeIds2.size} (was ${externalNodeIds.size})`);

  const analysis2 = analyzeGraph(plan2.nodes, externalNodeIds2, undefined);
  console.log(`\nAnalysis 2 (step 2):`);
  logAnalysis(analysis2, plan2);

  // ── Compare orderings ──
  console.log("\n=== Ordering comparison ===");
  const ops1 = analysis1.planNodes.map((n: any) => n.op);
  const ops2 = analysis2.planNodes.map((n: any) => n.op);
  let firstDiff = -1;
  for (let i = 0; i < Math.max(ops1.length, ops2.length); i++) {
    if (ops1[i] !== ops2[i]) { firstDiff = i; break; }
  }
  if (firstDiff === -1) {
    console.log("planNodes op sequences are IDENTICAL");
  } else {
    console.log(`planNodes op sequences DIFFER at position ${firstDiff}: ${ops1[firstDiff]} vs ${ops2[firstDiff]}`);
  }

  // Check ref kinds
  let refDiffs = 0;
  for (let i = 0; i < Math.min(analysis1.planNodes.length, analysis2.planNodes.length); i++) {
    const n1 = analysis1.planNodes[i];
    const n2 = analysis2.planNodes[i];
    for (let j = 0; j < Math.min(n1.inputs.length, n2.inputs.length); j++) {
      if (n1.inputs[j].kind !== n2.inputs[j].kind) {
        if (refDiffs < 5) {
          console.log(`  ref kind diff at node ${i} (${n1.op}) input ${j}: ${n1.inputs[j].kind} vs ${n2.inputs[j].kind}`);
        }
        refDiffs++;
      }
    }
  }
  console.log(`Total ref kind diffs: ${refDiffs}`);

  // Compare segments
  console.log("\n=== Segment comparison ===");
  console.log(`Analysis 1: ${analysis1.segments.length} segments`);
  console.log(`Analysis 2: ${analysis2.segments.length} segments`);
  for (let i = 0; i < Math.min(analysis1.segments.length, analysis2.segments.length); i++) {
    const s1 = analysis1.segments[i];
    const s2 = analysis2.segments[i];
    const k1 = s1.kind;
    const k2 = s2.kind;
    const n1 = s1.kind === "fused" ? s1.group.nodes.length : s1.nodes.length;
    const n2 = s2.kind === "fused" ? s2.group.nodes.length : s2.nodes.length;
    if (k1 !== k2 || n1 !== n2) {
      console.log(`  DIFF at segment ${i}: ${k1}(${n1}) vs ${k2}(${n2})`);
    }
  }

  process.exit(0);
}

function logAnalysis(analysis: any, plan: any) {
  console.log(`  planNodes: ${analysis.planNodes.length} nodes`);
  console.log(`  segments: ${analysis.segments.length}`);
  const segSummary = analysis.segments.map((s: any) => {
    const kind = s.kind;
    const count = kind === "fused" ? s.group.nodes.length : s.nodes.length;
    return `${kind}(${count})`;
  }).join(", ");
  console.log(`  layout: ${segSummary}`);

  // First 10 ops in planNodes
  const first10 = analysis.planNodes.slice(0, 10).map((n: any) => n.op).join(", ");
  console.log(`  first10: ${first10}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
