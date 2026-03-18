/**
 * Torchlette oracle: dump intermediate tensors for comparison with PyTorch.
 * Run: npx tsx tools/oracle-compare-torchlette.ts > /tmp/torchlette_oracle.json
 * Then: python3 -c "import json; a=json.load(open('/tmp/pytorch_oracle.json')); b=json.load(open('/tmp/torchlette_oracle.json')); [print(f'{k}: maxdiff={max(abs(x-y) for x,y in zip(a[k],b[k])):.2e}') if isinstance(a[k],list) else print(f'{k}: {a[k]:.6f} vs {b[k]:.6f} diff={abs(a[k]-b[k]):.2e}') for k in a if k in b]"
 */
import * as fs from "node:fs";
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false });
  const d = path.join(process.cwd(), "models", "gpt2");
  const tok = new GPT2Tokenizer();
  tok.load(
    JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
    fs
      .readFileSync(path.join(d, "merges.txt"), "utf-8")
      .split("\n")
      .filter((l) => l && !l.startsWith("#")),
  );

  const model = new GPT2WithLoRA(
    api,
    CONFIG,
    { rank: 64, alpha: 64 },
    "webgpu",
  );
  const buf = fs.readFileSync(path.join(d, "model.safetensors"));
  const hl = Number(
    new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true),
  );
  const hdr = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hl)));
  const w = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [n, m] of Object.entries(hdr) as [string, any][]) {
    if (n === "__metadata__" || m.dtype !== "F32") continue;
    const r = buf.subarray(
      8 + hl + m.data_offsets[0],
      8 + hl + m.data_offsets[1],
    );
    w.set(n.replace(/^transformer\./, ""), {
      data: new Float32Array(new Uint8Array(r).slice().buffer),
      shape: m.shape,
    });
  }
  model.loadBaseWeights(w);

  // Match PyTorch deterministic init: torch.manual_seed(42) per layer
  // For now, just set A=ones*0.01, B=zeros (simplest deterministic)
  const loraParams = model.getLoRAParameters();
  const runtime = api._runtime();
  for (let i = 0; i < loraParams.length; i += 2) {
    const loraA = loraParams[i];
    const aSize = loraA.shape.reduce((a: number, b: number) => a * b, 1);
    const aTensor = api.tensorFromArray(
      new Float32Array(aSize).fill(0.01),
      loraA.shape,
      { device: "webgpu" },
    );
    runtime.copy_(loraA._unwrap(), aTensor._unwrap());
  }
  await api.markStep();

  // We need to instrument the forward pass to capture intermediates.
  // Since we can't easily instrument the class, run the forward manually:
  const text = fs
    .readFileSync("node_modules/.cache/tinyshakespeare.txt", "utf-8")
    .slice(0, 5000);
  const tokens = tok.encode(text);
  const result: Record<string, number[] | number> = {};

  model.train(true);
  await api.beginStep();

  const input = api.tensorFromArray(tokens.slice(0, 128), [1, 128], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(tokens.slice(1, 129), [1, 128], {
    device: "webgpu",
  });

  // We can't easily intercept block outputs, so just capture loss + gradients
  const { loss, logits } = model.forwardWithLoss(input, target);
  const lossVal = await loss.item();
  result.loss = lossVal;

  // Capture logits[:20]
  const logitsData = await logits.cpu();
  result.logits = Array.from(logitsData.slice(0, 20));

  await loss.backward();

  // Capture gradients
  for (let i = 0; i < Math.min(12, loraParams.length / 2); i++) {
    const A = loraParams[i * 2];
    const B = loraParams[i * 2 + 1];
    const aGrad = A.grad ? await A.grad.cpu() : null;
    const bGrad = B.grad ? await B.grad.cpu() : null;
    if (aGrad) {
      result[`block${i}_A_grad_norm`] = Math.sqrt(
        Array.from(aGrad).reduce((s, v) => s + v * v, 0),
      );
      result[`block${i}_A_grad_first5`] = Array.from(aGrad.slice(0, 5));
    }
    if (bGrad) {
      result[`block${i}_B_grad_norm`] = Math.sqrt(
        Array.from(bGrad).reduce((s, v) => s + v * v, 0),
      );
      result[`block${i}_B_grad_first5`] = Array.from(bGrad.slice(0, 5));
    }
  }

  api.endStep();
  await api.markStep();

  console.log(JSON.stringify(result, null, 2));
  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
