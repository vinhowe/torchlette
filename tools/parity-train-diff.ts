/**
 * Multi-step weight-transfer PARITY trainer (torchlette side).
 *
 * Loads the SAME shared weights written by gen_parity_weights.py into a plain
 * torchlette GPT2, trains STEPS fp32 steps over the SAME window offsets with
 * identical AdamW (adamW=true, lr/wd) + grad clip, and writes per-step loss to
 * parity/tl.losses.json. Diffed against tools/diloco-pytorch/train_parity_pt.py.
 *
 * fp32, no autocast, no checkpointing — isolate kernel numerics, mirror PT.
 *
 * Env mirrors the PyTorch side: NUM_LAYERS NUM_HEADS EMBED_DIM SEQ_LEN
 * BATCH_SIZE STEPS LR WEIGHT_DECAY GRAD_CLIP PARITY_DIR LOCAL_TOKENS
 * WINDOW_OFFSETS.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { Adam } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";

const PARITY =
  process.env.PARITY_DIR ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/parity";
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
const STEPS = parseInt(process.env.STEPS ?? "60", 10);
const LR = parseFloat(process.env.LR ?? "5e-4");
const WD = parseFloat(process.env.WEIGHT_DECAY ?? "0.01");
const GRAD_CLIP = parseFloat(process.env.GRAD_CLIP ?? "1.0");
const VOCAB = 50257;

const log = (m: string) => console.error(`[tl-parity] ${m}`);

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

// canonical key -> torchlette named-parameter path (mirror parity-forward-diff)
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
    {
      vocabSize: VOCAB,
      blockSize: 1024,
      numLayers: L,
      numHeads: H,
      embedDim: E,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );
  const named = new Map<string, Tensor>(model.namedParameters());
  const paddedVocab = (model as { paddedVocabSize: number }).paddedVocabSize;

  // Load shared weights (wte zero-padded to padded vocab, same as fwd-diff).
  await api.beginStep();
  for (const key of Object.keys(manifest)) {
    const data = readF32(path.join(PARITY, `w.${key}.f32`));
    const tlParam = named.get(tlName(key));
    if (!tlParam) throw new Error(`torchlette missing param for ${key}`);
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
  log("weights loaded");

  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: LR, weightDecay: WD, adamW: true }, api);

  const toks = readU16(TOKENS);
  const offs = readI32(OFFSETS);

  const losses: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    // Slice the same windows the PyTorch side uses: offs[step*BATCH + b].
    const inp = new Int32Array(BATCH * SEQ);
    const tgt = new Int32Array(BATCH * SEQ);
    for (let b = 0; b < BATCH; b++) {
      const s = offs[step * BATCH + b]!;
      for (let t = 0; t < SEQ; t++) {
        inp[b * SEQ + t] = toks[s + t]!;
        tgt[b * SEQ + t] = toks[s + 1 + t]!;
      }
    }

    await api.beginStep();
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    const loss = api.tidy(() => {
      const l = model.forwardWithLoss(input, target, {
        useCheckpoint: false,
      }).loss;
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
    losses.push(lossVal);
    await loss.backward();
    if (GRAD_CLIP > 0) clipGradNorm_(api, params, GRAD_CLIP);
    opt.step();
    opt.zeroGrad();
    api.endStep();
    await api.markStep();

    if (step < 5 || step % 10 === 0 || step === STEPS - 1) {
      log(`step ${step}: loss=${lossVal.toFixed(6)}`);
    }
  }

  fs.writeFileSync(
    path.join(PARITY, "tl.losses.json"),
    JSON.stringify({ losses }),
  );
  log(`wrote ${losses.length} losses to ${path.join(PARITY, "tl.losses.json")}`);

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
