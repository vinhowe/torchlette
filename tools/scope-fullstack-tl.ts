/**
 * FULL-STACK multi-step parity trainer (torchlette side).
 *
 * Replicates the production WebGPUGPT2Trainer inner step EXACTLY — autocast
 * (f16) + gradient checkpointing + GradScaler + grad clip + AdamW — over the
 * SAME shared weights + window offsets as the PyTorch side
 * (tools/diloco-pytorch/fullstack_pt.py). Writes:
 *   parity/<OUT>.losses.json    : per-step loss trajectory
 *   parity/<GRAD_DIR>/g.<key>.f32 : step-0 grads (post-unscale, PRE-clip)
 *
 * Run lowered with TORCHLETTE_COMPILED_PLAN=0, compiled by default — the two
 * tl trajectories + the PyTorch one are then diffed 3-way.
 *
 * Env: NUM_LAYERS NUM_HEADS EMBED_DIM SEQ_LEN BATCH_SIZE STEPS LR WEIGHT_DECAY
 *      GRAD_CLIP USE_AUTOCAST(=1) CHECKPOINT(=1) USE_SCALER(=1)
 *      OUT(losses tag) GRAD_DIR PARITY_DIR LOCAL_TOKENS WINDOW_OFFSETS
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { storageTracker } from "../src/graph/storage-tracker";

const PARITY =
  process.env.PARITY_DIR ?? "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/parity";
const TOKENS =
  process.env.LOCAL_TOKENS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin";
const OFFSETS =
  process.env.WINDOW_OFFSETS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/window-offsets.i32";
const L = parseInt(process.env.NUM_LAYERS ?? "8", 10);
const H = parseInt(process.env.NUM_HEADS ?? "4", 10);
const E = parseInt(process.env.EMBED_DIM ?? "128", 10);
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH_SIZE ?? "8", 10);
const STEPS = parseInt(process.env.STEPS ?? "30", 10);
const LR = parseFloat(process.env.LR ?? "5e-4");
const WD = parseFloat(process.env.WEIGHT_DECAY ?? "0.01");
const GRAD_CLIP = parseFloat(process.env.GRAD_CLIP ?? "1.0");
const USE_AUTOCAST = process.env.USE_AUTOCAST !== "0";
const CHECKPOINT = process.env.CHECKPOINT !== "0";
const USE_SCALER = process.env.USE_SCALER !== "0";
const USE_SCOPE = process.env.USE_SCOPE !== "0";
const OUT = process.env.OUT ?? "tl";
const GRAD_DIR = process.env.GRAD_DIR ?? "tl_fs";
const VOCAB = 50257;

const log = (m: string) => console.error(`[tl-fs] ${m}`);

function readF32(p: string): Float32Array {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function readI32(p: string): Int32Array {
  const b = fs.readFileSync(p);
  return new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function readU16(p: string): Uint16Array {
  const b = fs.readFileSync(p);
  return new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2);
}
function tlName(key: string): string {
  if (key === "wte") return "wte.weight";
  if (key === "wpe") return "wpe.weight";
  if (key === "lnf.w") return "lnF.weight";
  if (key === "lnf.b") return "lnF.bias";
  const m = key.match(/^block\.(\d+)\.(.+)$/);
  if (!m) throw new Error(`bad key ${key}`);
  const [, i, rest] = m;
  const map: Record<string, string> = {
    "ln1.w": `h.${i}.ln1.weight`,
    "ln1.b": `h.${i}.ln1.bias`,
    "attn.qkv.w": `h.${i}.attn.cAttn.weight`,
    "attn.qkv.b": `h.${i}.attn.cAttn.bias`,
    "attn.proj.w": `h.${i}.attn.cProj.weight`,
    "attn.proj.b": `h.${i}.attn.cProj.bias`,
    "ln2.w": `h.${i}.ln2.weight`,
    "ln2.b": `h.${i}.ln2.bias`,
    "mlp.fc.w": `h.${i}.mlp.cFc.weight`,
    "mlp.fc.b": `h.${i}.mlp.cFc.bias`,
    "mlp.proj.w": `h.${i}.mlp.cProj.weight`,
    "mlp.proj.b": `h.${i}.mlp.cProj.bias`,
  };
  const out = map[rest!];
  if (!out) throw new Error(`bad key ${key}`);
  return out;
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "6000";

  const manifest: Record<string, number[]> = JSON.parse(
    fs.readFileSync(path.join(PARITY, "manifest.json"), "utf-8"),
  );

  const api = new Torchlette("webgpu", { enableFusion: true });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    { vocabSize: VOCAB, blockSize: 1024, numLayers: L, numHeads: H, embedDim: E, dropoutRate: 0 },
    { device: "webgpu" },
  );
  const named = new Map<string, Tensor>(model.namedParameters());
  const paddedVocab = (model as { paddedVocabSize: number }).paddedVocabSize;

  await api.beginStep();
  for (const key of Object.keys(manifest)) {
    const data = readF32(path.join(PARITY, `w.${key}.f32`));
    const tlParam = named.get(tlName(key));
    if (!tlParam) throw new Error(`missing param for ${key}`);
    let loadData: Float32Array = data;
    let loadShape = manifest[key]!;
    if (key === "wte" && paddedVocab > VOCAB) {
      const padded = new Float32Array(paddedVocab * E);
      padded.set(data);
      loadData = padded;
      loadShape = [paddedVocab, E];
    }
    api.copy_(
      tlParam,
      api.tensorFromArray(Array.from(loadData), loadShape, { device: "webgpu" }),
    );
  }
  api.endStep();
  await api.markStep();
  log(
    `weights loaded; autocast=${USE_AUTOCAST} checkpoint=${CHECKPOINT} scaler=${USE_SCALER} clip=${GRAD_CLIP} steps=${STEPS}`,
  );

  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: LR, weightDecay: WD, adamW: true }, api);
  const scaler = USE_SCALER ? new GradScaler(api, { initScale: 1024.0 }) : null;

  const toks = readU16(TOKENS);
  const offs = readI32(OFFSETS);
  const keys = Object.keys(manifest);

  const losses: number[] = [];
  const statHistory: { step: number; total: number; reachable: number }[] = [];
  for (let step = 0; step < STEPS; step++) {
    if (scaler) await scaler.resolveDeferred();

    const inp = new Int32Array(BATCH * SEQ);
    const tgt = new Int32Array(BATCH * SEQ);
    for (let b = 0; b < BATCH; b++) {
      const s = offs[step * BATCH + b]!;
      for (let t = 0; t < SEQ; t++) {
        inp[b * SEQ + t] = toks[s + t]!;
        tgt[b * SEQ + t] = toks[s + 1 + t]!;
      }
    }

    // SCOPE VARIANT: replace beginStep()/…/endStep()/markStep() ceremony with
    // a single async scope wrapping forward + backward + optimizer.step().
    // USE_SCOPE=0 falls back to the markStep ceremony (apples-to-apples
    // baseline through the SAME stats instrumentation).
    let lossVal = NaN;
    const body = async () => {
      const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], { device: "webgpu" });
      const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], { device: "webgpu" });
      const loss = api.tidy(() => {
        const fwd = () =>
          model.forwardWithLoss(input, target, { useCheckpoint: CHECKPOINT }).loss;
        const l = USE_AUTOCAST ? api.autocast(fwd) : fwd();
        // NOTE: no api.keep(l) here — under an open scope keep() escapes to
        // ROOT (permanent) and would leak one loss storage per step. tidy()'s
        // structural return-escape keeps l usable for item()/backward() inside
        // the scope; the scope reclaims it at close.
        if (!USE_SCOPE) api.keep(l);
        return l;
      });
      // loss.item() readback happens INSIDE the scope (provides a fence).
      lossVal = await loss.item();

      const backwardTarget = scaler ? scaler.scale(loss) : loss;
      await backwardTarget.backward();

      if (scaler) scaler.unscale_(opt);

      if (GRAD_CLIP > 0) clipGradNorm_(api, params, GRAD_CLIP);
      if (scaler) {
        scaler.step(opt);
        scaler.update();
      } else {
        opt.step();
      }
      opt.zeroGrad();
      input.dispose();
      target.dispose();
    };
    if (USE_SCOPE) {
      await api.scope(body);
    } else {
      await api.beginStep();
      await body();
      api.endStep();
      await api.markStep();
    }
    losses.push(lossVal);

    const st = storageTracker.stats();
    statHistory.push({ step, total: st.totalStorages, reachable: st.reachableStorages });

    if (step < 5 || step % 10 === 0 || step === STEPS - 1) {
      log(
        `step ${step}: loss=${lossVal.toFixed(6)} storages total=${st.totalStorages} reachable=${st.reachableStorages}`,
      );
    }
  }

  log(
    `STORAGE HISTORY: ${statHistory
      .map((s) => `${s.step}:${s.total}/${s.reachable}`)
      .join(" ")}`,
  );

  fs.writeFileSync(
    path.join(PARITY, `${OUT}.losses.json`),
    JSON.stringify({ losses }),
  );
  log(`wrote ${losses.length} losses to ${OUT}.losses.json`);

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  log(`STACK: ${(e as Error).stack}`);
  process.exit(1);
});
