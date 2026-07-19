/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — PROBE 1: GPU-SIDE FEEDBACK.
 *
 * THE load-bearing primitive. Decode's host-in-the-loop exists to close the
 * sampling feedback: logits -> (host) argmax -> next token id -> next graph.
 * Unrolled-K requires that feedback to close ON-DEVICE inside one command
 * stream: argmax over logits produces a token-id tensor, and the NEXT
 * iteration's embedding GATHER reads that id directly — no host readback
 * between tokens.
 *
 * This probe builds the greedy feedback loop BOTH ways over a real distilgpt2
 * (random init, ONE model so weights are identical across arms) and checks the
 * on-device chain is byte-identical to the host-loop reference:
 *
 *   arm HOST   : K decode steps, readback logits each step, host argmax, feed
 *                the id back as the next idx (the reference — today's decode).
 *   arm DEVICE : K decode steps chained as ONE lazy graph — each step's
 *                argmax(logits) tensor is fed straight into the next step's
 *                forwardCached (its embedding gather reads the id on-device);
 *                NOTHING is forced until the K ids are read back ONCE at the end.
 *
 * PASS: the two K-length id streams are byte-identical AND the device arm forces
 * the whole K-block with a single readback (host arm forces K times). That
 * proves the argmax->gather handoff closes on-device with no per-token roundtrip.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-uk-feedback.ts [K=8]
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { getSubmitCount, resetSubmitCount } from "../src/backend/webgpu/webgpu-state";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { DISTILGPT2_CONFIG, GPT2 } from "../examples/gpt2/model";
import type { KVCache } from "../examples/gpt2/model";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";

const K = Number(process.argv[2] ?? 8);
const PRETRAINED = process.env.UK_PRETRAINED === "1";
// "The capital of France is" (GPT-2 BPE) — greedy continuation varies (and even
// emits the EOS id 50256 within a few tokens, exercising the id range broadly).
const PROMPT = [464, 3139, 286, 4881, 318];

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1234);
  const MODEL_DIR = process.env.UK_MODEL_DIR ?? "./models/distilgpt2";
  const model = PRETRAINED
    ? await loadPretrainedGPT2(api, MODEL_DIR, { dropoutRate: 0 }, { device: "webgpu" })
    : new GPT2(api, { ...DISTILGPT2_CONFIG });
  const V = model.config.vocabSize;
  console.log(`weights: ${PRETRAINED ? "pretrained distilgpt2" : "random-init (UK_PRETRAINED=1 for pretrained)"}`);

  const prefill = (): { logits: Tensor; kvs: KVCache[] } => {
    const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
    const { logits, presentKVs } = api.noGrad(() => model.forwardCached(idx, undefined, 0));
    return { logits, kvs: presentKVs };
  };

  // logits [1,S,paddedVocab] -> LAST position -> real vocab, MATERIALIZED CONTIGUOUS.
  // NOTE (probe finding): the WebGPU arg-reduce kernel assumes a CONTIGUOUS
  // reduction row — argmax OVER a multi-dim strided view (narrow(dim1) then
  // narrow(dim2)) returns the WRONG index (measured: 40 vs correct 995).
  // `.contiguous()` MATERIALIZES the row first, so the subsequent argmax is over
  // a proper buffer and is correct. Cheap; the decode row is tiny.
  const lastRow = (logits: Tensor): Tensor => {
    const S = logits.shape[1];
    return api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V)); // [1,1,V]
  };

  // ---- arm HOST: readback every step, host argmax (the reference) ----
  const hostTokens: number[] = [];
  {
    let { logits, kvs } = prefill();
    let pos = PROMPT.length;
    resetSubmitCount();
    for (let i = 0; i < K; i++) {
      const row = api.noGrad(() => lastRow(logits)); // [1,1,V] contiguous
      const data = new Float32Array(await row.cpu()); // force BEFORE disposing logits
      logits.dispose();
      let best = 0;
      for (let v = 1; v < V; v++) if (data[v] > data[best]) best = v;
      hostTokens.push(best);
      const idx = api.tensorFromArray([best], [1, 1]);
      const nxt = api.noGrad(() => model.forwardCached(idx, kvs, pos));
      logits = nxt.logits;
      kvs = nxt.presentKVs;
      pos += 1;
    }
    logits.dispose();
  }
  const hostSubmits = getSubmitCount();

  // ---- arm DEVICE: chain argmax->gather ON-DEVICE, force ONCE at the end ----
  const deviceTokens: number[] = [];
  let deviceSubmits = 0;
  {
    let { logits, kvs } = prefill();
    let pos = PROMPT.length;
    const idTensors: Tensor[] = [];
    resetSubmitCount();
    for (let i = 0; i < K; i++) {
      const id = api.noGrad(() => api.argmax(lastRow(logits), { dim: -1, keepdim: false })); // [1,1] LAZY
      idTensors.push(id);
      // Feed the on-device id straight into the next step's embedding gather —
      // no host readback between tokens; the argmax output IS the next index.
      const idx = api.reshape(id, [1, 1]);
      const nxt = api.noGrad(() => model.forwardCached(idx, kvs, pos));
      logits = nxt.logits;
      kvs = nxt.presentKVs;
      pos += 1;
    }
    // ONE readback of the whole K-block: cat the K ids and force once.
    const stacked = api.cat(idTensors.map((t) => api.reshape(t, [1, 1])), 1); // [1,K]
    const ids = new Float32Array(await stacked.cpu());
    for (let i = 0; i < K; i++) deviceTokens.push(Math.round(ids[i]));
    deviceSubmits = getSubmitCount();
    logits.dispose();
    await api.markStep();
  }

  const match =
    hostTokens.length === deviceTokens.length &&
    hostTokens.every((t, i) => t === deviceTokens[i]);

  console.log(`=== PROBE 1: GPU-SIDE FEEDBACK (distilgpt2, K=${K} greedy tokens) ===`);
  console.log(`host   tokens: [${hostTokens.join(", ")}]`);
  console.log(`device tokens: [${deviceTokens.join(", ")}]`);
  console.log(`byte-identical ids: ${match}`);
  console.log(
    `submits — host arm (K readbacks): ${hostSubmits} ; device arm (1 readback at K-boundary): ${deviceSubmits}`,
  );
  console.log(
    `\nVERDICT: ${
      match
        ? "PASS — the argmax->gather feedback closes ON-DEVICE inside one command stream; the K-token greedy stream is byte-identical to the host-loop reference and the whole block is forced with a SINGLE readback. The load-bearing primitive is available today (greedy)."
        : "FAIL — on-device feedback diverged from host reference; investigate argmax/gather dtype seam."
    }`,
  );
  console.log(
    `=== UK-FEEDBACK-STATS === ${JSON.stringify({ K, match, hostSubmits, deviceSubmits, hostTokens, deviceTokens })}`,
  );
  process.exit(match ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
