/**
 * Systematic comparison of individual op gradients against PyTorch.
 * Tests each op used in GPT-2 to identify which ones have gradient differences.
 */

import { spawn } from "node:child_process";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { crossEntropy } from "../../src/nn/functional";

let initialized = false;

async function createApi(): Promise<Torchlette> {
  if (!initialized) {
    await initWebGPU();
    initialized = true;
  }
  return new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });
}

const PYTHON_PATH =
  process.env.TORCH_ORACLE_PYTHON ||
  "/Users/vin/dev/summer-2024-grant-project/torchlette/.venv/bin/python";

interface TensorPayload {
  shape: number[];
  values: number[];
}

interface OracleResult {
  ok: boolean;
  output?: TensorPayload;
  grads?: (TensorPayload | null)[];
  error?: string;
  caseName?: string;
}

async function callOracle(
  cases: Array<{
    caseName: string;
    op: string;
    inputs: TensorPayload[];
    options?: Record<string, unknown>;
  }>,
): Promise<OracleResult[]> {
  const payload = JSON.stringify({ cases });

  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_PATH, ["tools/torch_oracle/torch_oracle.py"]);

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Oracle failed: ${stderr}`));
        return;
      }
      try {
        const result = JSON.parse(stdout);
        resolve(result.results);
      } catch {
        reject(new Error(`Failed to parse oracle output: ${stdout}`));
      }
    });

    proc.stdin.write(payload);
    proc.stdin.end();
  });
}

function tensorPayload(
  data: Float32Array | number[],
  shape: number[],
): TensorPayload {
  return {
    shape,
    values: Array.from(data),
  };
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  }
  return maxDiff;
}

function relError(a: Float32Array, b: Float32Array): number {
  let maxRel = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1e-8);
    maxRel = Math.max(maxRel, Math.abs(a[i] - b[i]) / denom);
  }
  return maxRel;
}

interface TestResult {
  name: string;
  passed: boolean;
  forwardMaxDiff?: number;
  gradMaxDiffs?: number[];
  gradRelErrors?: number[];
  error?: string;
}

async function testMatmulBackward(): Promise<TestResult> {
  const api = await createApi();

  // Simple matmul: [2, 3] @ [3, 4] = [2, 4]
  const aData = new Float32Array([1, 2, 3, 4, 5, 6]);
  const bData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

  // Torchlette
  const a = api.tensorFromArray(aData, [2, 3], { requiresGrad: true });
  const b = api.tensorFromArray(bData, [3, 4], { requiresGrad: true });
  const c = a.matmul(b);
  const loss = c.sum();
  await loss.backward();

  const cTl = await c.cpu();
  if (!a.grad || !b.grad) {
    return { name: "matmul", passed: false, error: "Missing gradients" };
  }
  const aGradTl = await a.grad.cpu();
  const bGradTl = await b.grad.cpu();

  // PyTorch
  const [result] = await callOracle([
    {
      caseName: "matmul",
      op: "backward",
      inputs: [tensorPayload(aData, [2, 3]), tensorPayload(bData, [3, 4])],
      options: { op: "matmul" },
    },
  ]);

  if (!result.ok) {
    return { name: "matmul", passed: false, error: result.error };
  }

  const cPt = new Float32Array(result.output!.values);
  const aGradPt = new Float32Array(result.grads![0]!.values);
  const bGradPt = new Float32Array(result.grads![1]!.values);

  const forwardDiff = maxAbsDiff(cTl, cPt);
  const aGradDiff = maxAbsDiff(aGradTl, aGradPt);
  const bGradDiff = maxAbsDiff(bGradTl, bGradPt);

  const passed = forwardDiff < 1e-5 && aGradDiff < 1e-5 && bGradDiff < 1e-5;

  return {
    name: "matmul",
    passed,
    forwardMaxDiff: forwardDiff,
    gradMaxDiffs: [aGradDiff, bGradDiff],
    gradRelErrors: [relError(aGradTl, aGradPt), relError(bGradTl, bGradPt)],
  };
}

async function testAddBackward(): Promise<TestResult> {
  const api = await createApi();

  const aData = new Float32Array([1, 2, 3, 4]);
  const bData = new Float32Array([5, 6, 7, 8]);

  const a = api.tensorFromArray(aData, [2, 2], { requiresGrad: true });
  const b = api.tensorFromArray(bData, [2, 2], { requiresGrad: true });
  const c = a.add(b);
  const loss = c.sum();
  await loss.backward();

  const cTl = await c.cpu();
  if (!a.grad || !b.grad) {
    return { name: "add", passed: false, error: "Missing gradients" };
  }
  const aGradTl = await a.grad.cpu();
  const bGradTl = await b.grad.cpu();

  const [result] = await callOracle([
    {
      caseName: "add",
      op: "backward",
      inputs: [tensorPayload(aData, [2, 2]), tensorPayload(bData, [2, 2])],
      options: { op: "add" },
    },
  ]);

  if (!result.ok) {
    return { name: "add", passed: false, error: result.error };
  }

  const cPt = new Float32Array(result.output!.values);
  const aGradPt = new Float32Array(result.grads![0]!.values);
  const bGradPt = new Float32Array(result.grads![1]!.values);

  const forwardDiff = maxAbsDiff(cTl, cPt);
  const aGradDiff = maxAbsDiff(aGradTl, aGradPt);
  const bGradDiff = maxAbsDiff(bGradTl, bGradPt);

  return {
    name: "add",
    passed: forwardDiff < 1e-5 && aGradDiff < 1e-5 && bGradDiff < 1e-5,
    forwardMaxDiff: forwardDiff,
    gradMaxDiffs: [aGradDiff, bGradDiff],
  };
}

async function testMulBackward(): Promise<TestResult> {
  const api = await createApi();

  const aData = new Float32Array([1, 2, 3, 4]);
  const bData = new Float32Array([5, 6, 7, 8]);

  const a = api.tensorFromArray(aData, [2, 2], { requiresGrad: true });
  const b = api.tensorFromArray(bData, [2, 2], { requiresGrad: true });
  const c = a.mul(b);
  const loss = c.sum();
  await loss.backward();

  const cTl = await c.cpu();
  if (!a.grad || !b.grad) {
    return { name: "mul", passed: false, error: "Missing gradients" };
  }
  const aGradTl = await a.grad.cpu();
  const bGradTl = await b.grad.cpu();

  const [result] = await callOracle([
    {
      caseName: "mul",
      op: "backward",
      inputs: [tensorPayload(aData, [2, 2]), tensorPayload(bData, [2, 2])],
      options: { op: "mul" },
    },
  ]);

  if (!result.ok) {
    return { name: "mul", passed: false, error: result.error };
  }

  const cPt = new Float32Array(result.output!.values);
  const aGradPt = new Float32Array(result.grads![0]!.values);
  const bGradPt = new Float32Array(result.grads![1]!.values);

  const forwardDiff = maxAbsDiff(cTl, cPt);
  const aGradDiff = maxAbsDiff(aGradTl, aGradPt);
  const bGradDiff = maxAbsDiff(bGradTl, bGradPt);

  return {
    name: "mul",
    passed: forwardDiff < 1e-5 && aGradDiff < 1e-5 && bGradDiff < 1e-5,
    forwardMaxDiff: forwardDiff,
    gradMaxDiffs: [aGradDiff, bGradDiff],
  };
}

async function testReluBackward(): Promise<TestResult> {
  const api = await createApi();
    const xData = new Float32Array([-2, -1, 0, 1, 2, 3]);

    const x = api.tensorFromArray(xData, [2, 3], { requiresGrad: true });
    const y = x.relu();
    const loss = y.sum();
    await loss.backward();

    const yTl = await y.cpu();
    if (!x.grad) {
      return { name: "relu", passed: false, error: "Missing gradient" };
    }
    const xGradTl = await x.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "relu",
        op: "backward",
        inputs: [tensorPayload(xData, [2, 3])],
        options: { op: "relu" },
      },
    ]);

    if (!result.ok) {
      return { name: "relu", passed: false, error: result.error };
    }

    const yPt = new Float32Array(result.output!.values);
    const xGradPt = new Float32Array(result.grads![0]!.values);

    const forwardDiff = maxAbsDiff(yTl, yPt);
    const gradDiff = maxAbsDiff(xGradTl, xGradPt);

    return {
      name: "relu",
      passed: forwardDiff < 1e-5 && gradDiff < 1e-5,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [gradDiff],
    };
}

async function testGeluBackward(): Promise<TestResult> {
  const api = await createApi();
    const xData = new Float32Array([-2, -1, 0, 1, 2, 3]);

    const x = api.tensorFromArray(xData, [2, 3], { requiresGrad: true });
    const y = x.gelu();
    const loss = y.sum();
    await loss.backward();

    const yTl = await y.cpu();
    if (!x.grad) {
      return { name: "gelu", passed: false, error: "Missing gradient" };
    }
    const xGradTl = await x.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "gelu",
        op: "gelu_backward",
        inputs: [tensorPayload(xData, [2, 3])],
      },
    ]);

    if (!result.ok) {
      return { name: "gelu", passed: false, error: result.error };
    }

    const yPt = new Float32Array(result.output!.values);
    const xGradPt = new Float32Array(result.grads![0]!.values);

    const forwardDiff = maxAbsDiff(yTl, yPt);
    const gradDiff = maxAbsDiff(xGradTl, xGradPt);

    return {
      name: "gelu",
      passed: forwardDiff < 1e-4 && gradDiff < 1e-4,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [gradDiff],
      gradRelErrors: [relError(xGradTl, xGradPt)],
    };
}

async function testSoftmaxBackward(): Promise<TestResult> {
  const api = await createApi();
    const xData = new Float32Array([1, 2, 3, 4, 5, 6]);

    const x = api.tensorFromArray(xData, [2, 3], { requiresGrad: true });
    const y = x.softmax(-1);
    const loss = y.sum();
    await loss.backward();

    const yTl = await y.cpu();
    if (!x.grad) {
      return { name: "softmax", passed: false, error: "Missing gradient" };
    }
    const xGradTl = await x.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "softmax",
        op: "softmax_backward",
        inputs: [tensorPayload(xData, [2, 3])],
        options: { dim: -1 },
      },
    ]);

    if (!result.ok) {
      return { name: "softmax", passed: false, error: result.error };
    }

    const yPt = new Float32Array(result.output!.values);
    const xGradPt = new Float32Array(result.grads![0]!.values);

    const forwardDiff = maxAbsDiff(yTl, yPt);
    const gradDiff = maxAbsDiff(xGradTl, xGradPt);

    return {
      name: "softmax",
      passed: forwardDiff < 1e-5 && gradDiff < 1e-5,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [gradDiff],
      gradRelErrors: [relError(xGradTl, xGradPt)],
    };
}

async function testLayerNormBackward(): Promise<TestResult> {
  const api = await createApi();
    // [batch=2, seq=3, embed=4]
    const xData = new Float32Array([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
      22, 23, 24,
    ]);
    const wData = new Float32Array([1, 1, 1, 1]);
    const bData = new Float32Array([0, 0, 0, 0]);

    const x = api.tensorFromArray(xData, [2, 3, 4], { requiresGrad: true });
    const weight = api.tensorFromArray(wData, [4], { requiresGrad: true });
    const bias = api.tensorFromArray(bData, [4], { requiresGrad: true });

    const y = x.layernorm(weight, bias);
    const loss = y.sum();
    await loss.backward();

    const yTl = await y.cpu();
    if (!x.grad || !weight.grad || !bias.grad) {
      return { name: "layerNorm", passed: false, error: "Missing gradients" };
    }
    const xGradTl = await x.grad.cpu();
    const wGradTl = await weight.grad.cpu();
    const bGradTl = await bias.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "layer_norm",
        op: "layer_norm_backward",
        inputs: [
          tensorPayload(xData, [2, 3, 4]),
          tensorPayload(wData, [4]),
          tensorPayload(bData, [4]),
        ],
        options: { normalizedShape: [4] },
      },
    ]);

    if (!result.ok) {
      return { name: "layerNorm", passed: false, error: result.error };
    }

    const yPt = new Float32Array(result.output!.values);
    const xGradPt = new Float32Array(result.grads![0]!.values);
    const wGradPt = new Float32Array(result.grads![1]!.values);
    const bGradPt = new Float32Array(result.grads![2]!.values);

    const forwardDiff = maxAbsDiff(yTl, yPt);
    const xGradDiff = maxAbsDiff(xGradTl, xGradPt);
    const wGradDiff = maxAbsDiff(wGradTl, wGradPt);
    const bGradDiff = maxAbsDiff(bGradTl, bGradPt);

    const passed =
      forwardDiff < 1e-4 &&
      xGradDiff < 1e-4 &&
      wGradDiff < 1e-4 &&
      bGradDiff < 1e-4;

    return {
      name: "layerNorm",
      passed,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [xGradDiff, wGradDiff, bGradDiff],
      gradRelErrors: [
        relError(xGradTl, xGradPt),
        relError(wGradTl, wGradPt),
        relError(bGradTl, bGradPt),
      ],
    };
}

async function testEmbeddingBackward(): Promise<TestResult> {
  const api = await createApi();
    // vocab=5, embed=3
    const wData = new Float32Array([
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    ]);
    const indices = new Float32Array([0, 2, 1, 4]); // batch=2, seq=2

    const weight = api.tensorFromArray(wData, [5, 3], { requiresGrad: true });
    const idx = api.tensorFromArray(indices, [2, 2]);

    // Manual embedding lookup via gather
    const numElements = 4;
    const embeddingDim = 3;
    const flatInput = idx.reshape([numElements]);
    const expandedInput = flatInput
      .reshape([numElements, 1])
      .expand([numElements, embeddingDim])
      .contiguous();
    const gathered = weight.gather(expandedInput, { dim: 0 });
    const output = gathered.reshape([2, 2, 3]);

    const loss = output.sum();
    await loss.backward();

    const outTl = await output.cpu();
    if (!weight.grad) {
      return { name: "embedding", passed: false, error: "Missing gradient" };
    }
    const wGradTl = await weight.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "embedding",
        op: "embedding_backward",
        inputs: [tensorPayload(wData, [5, 3]), tensorPayload(indices, [2, 2])],
      },
    ]);

    if (!result.ok) {
      return { name: "embedding", passed: false, error: result.error };
    }

    const outPt = new Float32Array(result.output!.values);
    const wGradPt = new Float32Array(result.grads![0]!.values);

    const forwardDiff = maxAbsDiff(outTl, outPt);
    const gradDiff = maxAbsDiff(wGradTl, wGradPt);

    return {
      name: "embedding",
      passed: forwardDiff < 1e-5 && gradDiff < 1e-5,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [gradDiff],
    };
}

async function testCrossEntropyBackward(): Promise<TestResult> {
  const api = await createApi();
    // [batch=2, classes=4]
    const logitsData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const targetsData = new Float32Array([2, 0]); // class indices

    const logits = api.tensorFromArray(logitsData, [2, 4], {
      requiresGrad: true,
    });
    const targets = api.tensorFromArray(targetsData, [2]);

    const loss = crossEntropy(api, logits, targets);
    await loss.backward();

    const lossTl = await loss.cpu();
    if (!logits.grad) {
      return { name: "crossEntropy", passed: false, error: "Missing gradient" };
    }
    const logitsGradTl = await logits.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "cross_entropy",
        op: "cross_entropy_backward",
        inputs: [
          tensorPayload(logitsData, [2, 4]),
          tensorPayload(targetsData, [2]),
        ],
      },
    ]);

    if (!result.ok) {
      return { name: "crossEntropy", passed: false, error: result.error };
    }

    const lossPt = new Float32Array(result.output!.values);
    const logitsGradPt = new Float32Array(result.grads![0]!.values);

    const forwardDiff = Math.abs(lossTl[0] - lossPt[0]);
    const gradDiff = maxAbsDiff(logitsGradTl, logitsGradPt);

    return {
      name: "crossEntropy",
      passed: forwardDiff < 1e-5 && gradDiff < 1e-5,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [gradDiff],
      gradRelErrors: [relError(logitsGradTl, logitsGradPt)],
    };
}

async function testAttentionBackward(): Promise<TestResult> {
  const api = await createApi();
    // [batch=1, heads=2, seq=3, head_dim=4]
    // Use fixed seed for reproducibility
    const qData = new Float32Array([
      0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3, 0.4, -0.5, 0.1, -0.2, 0.3,
      -0.4, 0.5, -0.1, 0.2, -0.3, 0.4, -0.5, 0.1, -0.2, 0.3, -0.4,
    ]);
    const kData = new Float32Array([
      0.2, -0.1, 0.4, -0.3, 0.6, -0.2, 0.1, -0.4, 0.3, -0.6, 0.2, -0.1, 0.4,
      -0.3, 0.6, -0.2, 0.1, -0.4, 0.3, -0.6, 0.2, -0.1, 0.4, -0.3,
    ]);
    const vData = new Float32Array([
      0.3, -0.3, 0.5, -0.2, 0.7, -0.3, 0.0, -0.5, 0.2, -0.7, 0.3, -0.0, 0.5,
      -0.2, 0.7, -0.3, 0.0, -0.5, 0.2, -0.7, 0.3, -0.0, 0.5, -0.2,
    ]);

    const shape = [1, 2, 3, 4];
    const q = api.tensorFromArray(qData, shape, { requiresGrad: true });
    const k = api.tensorFromArray(kData, shape, { requiresGrad: true });
    const v = api.tensorFromArray(vData, shape, { requiresGrad: true });

    const scale = 1.0 / Math.sqrt(4);

    // Attention: softmax(Q @ K^T * scale) @ V
    const kT = k.transpose({ dim0: 2, dim1: 3 }).contiguous();
    const scores = q.matmul(kT).mul(api.tensorFromArray([scale], []));

    // Causal mask
    const seqLen = 3;
    const maskData = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        maskData[i * seqLen + j] = j > i ? -Infinity : 0;
      }
    }
    const mask = api.tensorFromArray(maskData, [seqLen, seqLen]);
    const maskedScores = scores.add(mask);

    const attn = maskedScores.softmax(-1);
    const output = attn.matmul(v);

    const loss = output.sum();
    await loss.backward();

    const outTl = await output.cpu();
    if (!q.grad || !k.grad || !v.grad) {
      return { name: "attention", passed: false, error: "Missing gradients" };
    }
    const qGradTl = await q.grad.cpu();
    const kGradTl = await k.grad.cpu();
    const vGradTl = await v.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "attention",
        op: "attention_backward",
        inputs: [
          tensorPayload(qData, shape),
          tensorPayload(kData, shape),
          tensorPayload(vData, shape),
        ],
        options: { scale, causal: true },
      },
    ]);

    if (!result.ok) {
      return { name: "attention", passed: false, error: result.error };
    }

    const outPt = new Float32Array(result.output!.values);
    const qGradPt = new Float32Array(result.grads![0]!.values);
    const kGradPt = new Float32Array(result.grads![1]!.values);
    const vGradPt = new Float32Array(result.grads![2]!.values);

    const forwardDiff = maxAbsDiff(outTl, outPt);
    const qGradDiff = maxAbsDiff(qGradTl, qGradPt);
    const kGradDiff = maxAbsDiff(kGradTl, kGradPt);
    const vGradDiff = maxAbsDiff(vGradTl, vGradPt);

    const passed =
      forwardDiff < 1e-4 &&
      qGradDiff < 1e-4 &&
      kGradDiff < 1e-4 &&
      vGradDiff < 1e-4;

    return {
      name: "attention",
      passed,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [qGradDiff, kGradDiff, vGradDiff],
      gradRelErrors: [
        relError(qGradTl, qGradPt),
        relError(kGradTl, kGradPt),
        relError(vGradTl, vGradPt),
      ],
    };
}

async function testLinearBackward(): Promise<TestResult> {
  const api = await createApi();
    // x: [batch=2, in=3], weight: [out=4, in=3], bias: [out=4]
    const xData = new Float32Array([1, 2, 3, 4, 5, 6]);
    const wData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const bData = new Float32Array([0.1, 0.2, 0.3, 0.4]);

    const x = api.tensorFromArray(xData, [2, 3], { requiresGrad: true });
    const w = api.tensorFromArray(wData, [4, 3], { requiresGrad: true });
    const b = api.tensorFromArray(bData, [4], { requiresGrad: true });

    // Linear: x @ W^T + b
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const y = x.matmul(wT).add(b);

    const loss = y.sum();
    await loss.backward();

    const yTl = await y.cpu();
    if (!x.grad || !w.grad || !b.grad) {
      return { name: "linear", passed: false, error: "Missing gradients" };
    }
    const xGradTl = await x.grad.cpu();
    const wGradTl = await w.grad.cpu();
    const bGradTl = await b.grad.cpu();

    const [result] = await callOracle([
      {
        caseName: "linear",
        op: "linear_backward",
        inputs: [
          tensorPayload(xData, [2, 3]),
          tensorPayload(wData, [4, 3]),
          tensorPayload(bData, [4]),
        ],
      },
    ]);

    if (!result.ok) {
      return { name: "linear", passed: false, error: result.error };
    }

    const yPt = new Float32Array(result.output!.values);
    const xGradPt = new Float32Array(result.grads![0]!.values);
    const wGradPt = new Float32Array(result.grads![1]!.values);
    const bGradPt = new Float32Array(result.grads![2]!.values);

    const forwardDiff = maxAbsDiff(yTl, yPt);
    const xGradDiff = maxAbsDiff(xGradTl, xGradPt);
    const wGradDiff = maxAbsDiff(wGradTl, wGradPt);
    const bGradDiff = maxAbsDiff(bGradTl, bGradPt);

    const passed =
      forwardDiff < 1e-4 &&
      xGradDiff < 1e-4 &&
      wGradDiff < 1e-4 &&
      bGradDiff < 1e-4;

    return {
      name: "linear",
      passed,
      forwardMaxDiff: forwardDiff,
      gradMaxDiffs: [xGradDiff, wGradDiff, bGradDiff],
      gradRelErrors: [
        relError(xGradTl, xGradPt),
        relError(wGradTl, wGradPt),
        relError(bGradTl, bGradPt),
      ],
    };
}

async function main() {
  console.log("=== Systematic Op Gradient Comparison vs PyTorch ===\n");

  const tests = [
    { name: "add", fn: testAddBackward },
    { name: "mul", fn: testMulBackward },
    { name: "matmul", fn: testMatmulBackward },
    { name: "relu", fn: testReluBackward },
    { name: "gelu", fn: testGeluBackward },
    { name: "softmax", fn: testSoftmaxBackward },
    { name: "layerNorm", fn: testLayerNormBackward },
    { name: "embedding", fn: testEmbeddingBackward },
    { name: "crossEntropy", fn: testCrossEntropyBackward },
    { name: "linear", fn: testLinearBackward },
    { name: "attention", fn: testAttentionBackward },
  ];

  const results: TestResult[] = [];

  for (const test of tests) {
    process.stdout.write(`Testing ${test.name}... `);
    try {
      const result = await test.fn();
      results.push(result);

      if (result.passed) {
        console.log("✓ PASS");
      } else {
        console.log("✗ FAIL");
      }
    } catch (e) {
      const err = e as Error;
      results.push({
        name: test.name,
        passed: false,
        error: err.message,
      });
      console.log(`✗ ERROR: ${err.message}`);
    }
  }

  console.log("\n=== Summary ===\n");

  const passed = results.filter((r) => r.passed);
  const failed = results.filter((r) => !r.passed);

  console.log(`Passed: ${passed.length}/${results.length}`);

  if (failed.length > 0) {
    console.log("\nFailed tests:");
    for (const r of failed) {
      console.log(`\n  ${r.name}:`);
      if (r.error) {
        console.log(`    Error: ${r.error}`);
      } else {
        console.log(
          `    Forward max diff: ${r.forwardMaxDiff?.toExponential(3)}`,
        );
        if (r.gradMaxDiffs) {
          console.log(
            `    Grad max diffs: ${r.gradMaxDiffs.map((d) => d.toExponential(3)).join(", ")}`,
          );
        }
        if (r.gradRelErrors) {
          console.log(
            `    Grad rel errors: ${r.gradRelErrors.map((d) => (d * 100).toFixed(2) + "%").join(", ")}`,
          );
        }
      }
    }
  }

  console.log("\nDetailed results:");
  for (const r of results) {
    console.log(`\n  ${r.name}: ${r.passed ? "PASS" : "FAIL"}`);
    if (r.forwardMaxDiff !== undefined) {
      console.log(`    Forward max diff: ${r.forwardMaxDiff.toExponential(3)}`);
    }
    if (r.gradMaxDiffs) {
      console.log(
        `    Grad max diffs: ${r.gradMaxDiffs.map((d) => d.toExponential(3)).join(", ")}`,
      );
    }
    if (r.gradRelErrors) {
      console.log(
        `    Grad rel errors: ${r.gradRelErrors.map((d) => (d * 100).toFixed(4) + "%").join(", ")}`,
      );
    }
  }

  process.exit(failed.length > 0 ? 1 : 0);
}

main().catch(console.error);
