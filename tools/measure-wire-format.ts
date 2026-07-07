/**
 * Measure the JSON wire format for real training plans.
 *
 * Wraps the e2e test's WebSocket transport to capture every plan flowing
 * through, then dumps a detailed breakdown: per-field, per-category, per-op.
 *
 * Usage:
 *   1. Start server: npx tsx examples/remote-training-demo/server.ts --port 9882
 *   2. Run this:    npx tsx tools/measure-wire-format.ts
 */

import WebSocket from "ws";
import { gzipSync } from "node:zlib";
import { nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatch,
  setTransitionMatrices,
  VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";
import type {
  ExecuteParams,
  ExecuteResult,
  DownloadParams,
  DownloadResult,
  ReadScalarParams,
  ReadScalarResult,
  UploadParams,
  UploadResult,
  ReleaseParams,
  ReleaseResult,
} from "../src/remote/rpc";
import type { Transport } from "../src/remote/client-engine";
import { createRemoteEngine } from "../src/remote/client-engine";
import type { SerializedPlan } from "../src/remote/wire";

// ============================================================================
// WebSocket transport (same as test-remote-e2e.ts) + plan capture
// ============================================================================

class CapturingNodeTransport implements Transport {
  private ws!: WebSocket;
  private nextId = 1;
  private pending = new Map<number, { resolve: (r: any) => void; reject: (e: Error) => void }>();
  sessionId = "";
  capturedPlans: { wire: SerializedPlan; jsonBytes: number; jsonStr: string }[] = [];

  async connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);
      let helloed = false;
      this.ws.on("message", (data: Buffer | string, isBinary: boolean) => {
        if (!isBinary) {
          const msg = JSON.parse(data.toString());
          if (!helloed && msg.id === 0) {
            this.sessionId = msg.result.sessionId;
            helloed = true;
            resolve();
            return;
          }
          const p = this.pending.get(msg.id);
          if (p) {
            this.pending.delete(msg.id);
            if (msg.error) p.reject(new Error(msg.error.message));
            else p.resolve(msg.result);
          }
        }
      });
      this.ws.on("error", reject);
    });
  }

  private rpc<T>(method: string, params: unknown): Promise<T> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async execute(params: ExecuteParams): Promise<ExecuteResult> {
    // Capture before forwarding
    const jsonStr = JSON.stringify(params.plan);
    this.capturedPlans.push({
      wire: params.plan,
      jsonBytes: jsonStr.length,
      jsonStr,
    });
    return this.rpc("execute", params);
  }
  upload(params: UploadParams): Promise<UploadResult> { return this.rpc("upload", params); }
  download(params: DownloadParams): Promise<DownloadResult> { return this.rpc("download", params); }
  readScalar(params: ReadScalarParams): Promise<ReadScalarResult> { return this.rpc("readScalar", params); }
  release(params: ReleaseParams): Promise<ReleaseResult> { return this.rpc("release", params); }

  close() { this.ws.close(); }
}

// ============================================================================
// Field-level analysis
// ============================================================================

function fmtBytes(b: number): string {
  if (b < 1024) return `${b}B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / (1024 * 1024)).toFixed(2)}MB`;
}

function jsonSize(v: unknown): number {
  return JSON.stringify(v).length;
}

interface PlanCapture {
  wire: SerializedPlan;
  jsonBytes: number;
  jsonStr: string;
}

function analyzePlan(p: PlanCapture, label: string): void {
  console.log(`\n${"=".repeat(78)}`);
  console.log(`${label}: ${p.wire.nodes.length} nodes, ${fmtBytes(p.jsonBytes)} JSON`);
  console.log("=".repeat(78));

  // Top-level fields
  const versionSize = jsonSize({ version: p.wire.version });
  const externalHandlesSize = jsonSize({ externalHandles: p.wire.externalHandles });
  const outputNodesSize = jsonSize({ outputNodes: p.wire.outputNodes });
  const nodesSize = jsonSize({ nodes: p.wire.nodes });

  console.log(`\nTop-level fields:`);
  console.log(`  version:         ${fmtBytes(versionSize)}`);
  console.log(`  externalHandles: ${fmtBytes(externalHandlesSize)} (${p.wire.externalHandles.length} entries)`);
  console.log(`  outputNodes:     ${fmtBytes(outputNodesSize)} (${p.wire.outputNodes?.length ?? 0} indices)`);
  console.log(`  nodes:           ${fmtBytes(nodesSize)}`);

  // Per-field across all nodes
  let opBytes = 0, inputsBytes = 0, shapeBytes = 0, dtypeBytes = 0;
  let deviceBytes = 0, payloadBytes = 0, moduleBytes = 0, idxBytes = 0;
  let payloadInlineBytes = 0;

  for (const node of p.wire.nodes) {
    opBytes += jsonSize({ op: node.op });
    inputsBytes += jsonSize({ inputs: node.inputs });
    shapeBytes += jsonSize({ shape: node.shape });
    dtypeBytes += jsonSize({ dtype: node.dtype });
    deviceBytes += jsonSize({ device: node.device });
    idxBytes += jsonSize({ idx: node.idx });
    if (node.payload !== undefined) {
      const ps = jsonSize({ payload: node.payload });
      payloadBytes += ps;
      if (JSON.stringify(node.payload).includes("__inlineTensor")) {
        payloadInlineBytes += ps;
      }
    }
    if (node.module !== undefined) moduleBytes += jsonSize({ module: node.module });
  }

  const total = opBytes + inputsBytes + shapeBytes + dtypeBytes + deviceBytes + payloadBytes + moduleBytes + idxBytes;
  const fields: [string, number][] = [
    ["inputs", inputsBytes], ["payload", payloadBytes], ["op", opBytes],
    ["shape", shapeBytes], ["module", moduleBytes], ["device", deviceBytes],
    ["dtype", dtypeBytes], ["idx", idxBytes],
  ];
  fields.sort((a, b) => b[1] - a[1]);

  console.log(`\nPer-field bytes (sum across all ${p.wire.nodes.length} nodes):`);
  for (const [name, bytes] of fields) {
    if (bytes === 0) continue;
    const pct = ((bytes / total) * 100).toFixed(1);
    console.log(`  ${name.padEnd(10)} ${fmtBytes(bytes).padStart(8)}  ${pct}%`);
  }
  console.log(`  ${"TOTAL".padEnd(10)} ${fmtBytes(total).padStart(8)}`);

  if (payloadInlineBytes > 0) {
    console.log(`  (of payload, ${fmtBytes(payloadInlineBytes)} is inline tensor data)`);
  }

  // Per-op breakdown
  const opCounts = new Map<string, { count: number; bytes: number }>();
  for (const node of p.wire.nodes) {
    const e = opCounts.get(node.op) ?? { count: 0, bytes: 0 };
    e.count++;
    e.bytes += jsonSize(node);
    opCounts.set(node.op, e);
  }
  const ops = [...opCounts.entries()].sort((a, b) => b[1].bytes - a[1].bytes).slice(0, 12);
  console.log(`\nTop ops by total bytes:`);
  for (const [op, { count, bytes }] of ops) {
    const avg = Math.round(bytes / count);
    const pct = ((bytes / total) * 100).toFixed(1);
    console.log(`  ${op.padEnd(28)} ${String(count).padStart(4)}x  ${fmtBytes(bytes).padStart(8)} (${avg}B avg, ${pct}%)`);
  }

  // Sample three nodes from different positions
  console.log(`\nSample nodes:`);
  for (const idx of [0, Math.floor(p.wire.nodes.length / 2), p.wire.nodes.length - 1]) {
    const node = p.wire.nodes[idx];
    const s = JSON.stringify(node);
    const trunc = s.length > 200 ? s.slice(0, 197) + "..." : s;
    console.log(`  #${idx} (${jsonSize(node)}B): ${trunc}`);
  }

  // Input ref kind breakdown
  let pendingRefs = 0, materializedRefs = 0, scalarRefs = 0;
  for (const node of p.wire.nodes) {
    for (const inp of node.inputs) {
      if (inp.kind === "pending") pendingRefs++;
      else if (inp.kind === "materialized") materializedRefs++;
      else scalarRefs++;
    }
  }
  console.log(`\nInput refs: ${pendingRefs} pending, ${materializedRefs} materialized, ${scalarRefs} scalar`);

  // Distinct value cardinalities — what's worth a dictionary
  const distinctOps = new Set(p.wire.nodes.map((n) => n.op));
  const distinctShapes = new Set(p.wire.nodes.map((n) => JSON.stringify(n.shape)));
  const distinctDtypes = new Set(p.wire.nodes.map((n) => n.dtype));
  const distinctModules = new Set(p.wire.nodes.map((n) => n.module).filter((m) => m !== undefined));
  console.log(`\nDistinct values:`);
  console.log(`  ops:     ${distinctOps.size}`);
  console.log(`  shapes:  ${distinctShapes.size}`);
  console.log(`  dtypes:  ${distinctDtypes.size}`);
  console.log(`  modules: ${distinctModules.size}`);

  // Compression
  const gz = gzipSync(p.jsonStr).length;
  console.log(`\nCompressed:`);
  console.log(`  raw:    ${fmtBytes(p.jsonBytes)}`);
  console.log(`  gzip:   ${fmtBytes(gz)} (${((gz / p.jsonBytes) * 100).toFixed(0)}%)`);
}

async function measureCpuCosts(p: PlanCapture, label: string): Promise<void> {
  console.log(`\n${"=".repeat(78)}`);
  console.log(`CPU costs for ${label} (${p.wire.nodes.length} nodes)`);
  console.log("=".repeat(78));

  const ITERS = 50;

  // JSON.stringify
  let t = performance.now();
  for (let i = 0; i < ITERS; i++) JSON.stringify(p.wire);
  const stringifyMs = (performance.now() - t) / ITERS;

  // JSON.parse
  t = performance.now();
  for (let i = 0; i < ITERS; i++) JSON.parse(p.jsonStr);
  const parseMs = (performance.now() - t) / ITERS;

  // gzip
  t = performance.now();
  for (let i = 0; i < ITERS; i++) gzipSync(p.jsonStr);
  const gzipMs = (performance.now() - t) / ITERS;

  // deserializePlan (tree walk into LazyIRNode objects)
  const { deserializePlan } = await import("../src/remote/serialize");
  t = performance.now();
  for (let i = 0; i < ITERS; i++) {
    deserializePlan(p.wire, { resolveHandle: () => ({} as any) });
  }
  const deserializeMs = (performance.now() - t) / ITERS;

  console.log(`  JSON.stringify:   ${stringifyMs.toFixed(2)}ms`);
  console.log(`  JSON.parse:       ${parseMs.toFixed(2)}ms`);
  console.log(`  gzip compress:    ${gzipMs.toFixed(2)}ms`);
  console.log(`  deserializePlan:  ${deserializeMs.toFixed(2)}ms`);
  console.log(`  TOTAL encode:     ${(stringifyMs + gzipMs).toFixed(2)}ms (client)`);
  console.log(`  TOTAL decode:     ${(parseMs + deserializeMs).toFixed(2)}ms (server)`);
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const SERVER_URL = process.env.SERVER_URL ?? "ws://localhost:9882/ws";
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 1 + 1, S = 10, B = 128;

  const transport = new CapturingNodeTransport();
  await transport.connect(SERVER_URL);

  const engine = createRemoteEngine(transport);
  const api = engine.torch;
  api.manualSeed(42);
  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-3 });

  await engine.preUpload(m.parameters());

  // 3 steps so we have a clean steady-state to analyze
  for (let step = 0; step < 3; step++) {
    await api.beginStep();
    const batch = generateBatch({ seqLen: S, batchSize: B });
    const tok = api.tensorFromArray(batch.tokens, [B, S], { dtype: "i32" });
    const tgt = api.tensorFromArray(batch.targets as number[], [B * (S - 1)], { dtype: "i32" });

    const loss = api.tidy(() => {
      const fwd = m.forward(tok);
      const logits = fwd.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
      const l = crossEntropy(api, logits, tgt);
      api.keep(l);
      return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
    loss.dispose();
    o.step(); o.zeroGrad();
    await api.endStep();
    await engine.markStep([...o.getAllKeepTensors(), ...m.persistentTensors()]);
  }

  console.log(`\nCaptured ${transport.capturedPlans.length} plans across 3 steps:`);
  for (let i = 0; i < transport.capturedPlans.length; i++) {
    const p = transport.capturedPlans[i];
    console.log(`  [${i}] ${p.wire.nodes.length} nodes  ${fmtBytes(p.jsonBytes)}`);
  }

  // Step 0 has setup plans; steady state starts around step 1-2.
  // Analyze the last 2 plans (one forceAllMerged + one forceAllPending from final step).
  const lastTwo = transport.capturedPlans.slice(-2);
  for (let i = 0; i < lastTwo.length; i++) {
    analyzePlan(lastTwo[i], `Steady-state plan ${i + 1}/2 (final step)`);
  }
  for (let i = 0; i < lastTwo.length; i++) {
    await measureCpuCosts(lastTwo[i], `plan ${i + 1}/2`);
  }

  // Steady-state totals
  console.log(`\n${"=".repeat(78)}`);
  console.log(`Steady-state per-step totals:`);
  console.log("=".repeat(78));
  let totalBytes = 0;
  let totalGz = 0;
  for (const p of lastTwo) {
    totalBytes += p.jsonBytes;
    totalGz += gzipSync(p.jsonStr).length;
  }
  console.log(`  ${lastTwo.length} plan(s) per step`);
  console.log(`  ${fmtBytes(totalBytes)} JSON per step`);
  console.log(`  ${fmtBytes(totalGz)} gzip'd per step (${((totalGz / totalBytes) * 100).toFixed(0)}% of raw)`);

  transport.close();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
