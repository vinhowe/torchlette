/**
 * In-browser SAE steering engine (Web Worker): WebGPU init, Gemma-2-2B weight
 * streaming, Gemma Scope SAE loading, feature inspection, and steered generation
 * — all off the main thread.
 *
 * Protocol (worker ← main):
 *   { type: "load", modelId, saeBaseUrl, weightDtype? }
 *   { type: "inspect", id, prompt, topK? }
 *   { type: "generate", id, prompt, steer, maxNewTokens?, temperature? }
 *       steer: { feature, alpha }[]
 * (worker → main):
 *   { type: "progress", loaded, total, status }
 *   { type: "loaded", modelId, weightDtype, numLayers, hiddenSize, saeLayer,
 *            numFeatures, neuronpediaSaeId, tapeFlagSet, tapeOn }
 *   { type: "tensor", ev }
 *   { type: "features", id, report }            (FeatureReport)
 *   { type: "delta", id, delta } | { type: "replace", id, text }
 *   { type: "done", id, stats }
 *   { type: "error", id?, error }
 */

import "./tape-flag"; // MUST be first: sets TORCHLETTE_STEP_TAPE before torchlette evaluates
import { AutoTokenizer } from "@huggingface/transformers";
import { STEP_TAPE_REPLAY } from "torchlette";
import {
  generateChat,
  loadGemma2FromUrl,
  type Gemma2,
} from "gemma2-browser";
import { GemmaScopeSAE, type SAEConfig, type SAEParams } from "gemma-scope-sae";
import {
  getGpuUncapturedErrorCount,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  Torchlette,
} from "torchlette";
import {
  inspectFeatures,
  makeSAEResidualHook,
  type SteerSpec,
} from "./sae-steering";

let api: Torchlette | null = null;
let model: Gemma2 | null = null;
let sae: GemmaScopeSAE | null = null;
let tokenizerLike:
  | {
      encode: (t: string) => number[];
      decode: (ids: number[], o?: { skip_special_tokens?: boolean }) => string;
    }
  | null = null;

const post = (msg: Record<string, unknown>) => {
  (self as unknown as Worker).postMessage(msg);
};

function heapMB(): string {
  const m = (performance as unknown as { memory?: { usedJSHeapSize: number } })
    .memory;
  return m ? ` · heap ${(m.usedJSHeapSize / 1e9).toFixed(1)}GB` : "";
}

/** Fetch the SAE .bin files from a static base URL into SAEParams. */
async function loadSAE(
  baseUrl: string,
  onStatus: (s: string) => void,
): Promise<GemmaScopeSAE> {
  onStatus("Loading SAE manifest…");
  const manifest = await (await fetch(`${baseUrl}/sae.json`)).json();
  const F = manifest.numFeatures as number;
  const d = manifest.dModel as number;
  const rd = async (name: string, expect: number): Promise<Float32Array> => {
    onStatus(`Loading SAE ${name}…`);
    const buf = await (
      await fetch(`${baseUrl}/${manifest.files[name]}`)
    ).arrayBuffer();
    const a = new Float32Array(buf);
    if (a.length !== expect)
      throw new Error(`SAE ${name}: ${a.length} floats, expected ${expect}`);
    return a;
  };
  const params: SAEParams = {
    W_enc: await rd("W_enc", d * F),
    b_enc: await rd("b_enc", F),
    W_dec: await rd("W_dec", F * d),
    b_dec: await rd("b_dec", d),
    threshold: await rd("threshold", F),
  };
  const config: SAEConfig = {
    dModel: d,
    numFeatures: F,
    layer: manifest.layer,
    neuronpediaSaeId: manifest.neuronpediaSaeId,
  };
  if (!api) throw new Error("api not ready");
  return GemmaScopeSAE.load(api, config, params, { dtype: "f32" });
}

async function handleLoad(
  modelId: string,
  saeBaseUrl: string,
  requestedDtype?: "f16" | "f32",
) {
  if (!("gpu" in self.navigator)) {
    throw new Error(
      "WebGPU not available (need Chrome/Edge 121+ or Safari 18+).",
    );
  }
  const adapter = await (
    self.navigator as unknown as {
      gpu: { requestAdapter(): Promise<{ features: Set<string> } | null> };
    }
  ).gpu.requestAdapter();
  const weightDtype: "f16" | "f32" =
    requestedDtype ?? (adapter?.features?.has("shader-f16") ? "f16" : "f32");

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  api = new Torchlette("webgpu", { enableFusion: true });

  const device = getWebGPUDevice() as unknown as {
    lost?: Promise<{ reason: string; message: string }>;
    limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number };
  } | null;
  device?.lost?.then((info) =>
    post({
      type: "error",
      error: `GPU device lost (${info.reason}): ${info.message}`,
    }),
  );
  const limits = device?.limits;
  post({
    type: "progress",
    loaded: 0,
    total: 0,
    status:
      `Loading tokenizer… (weights as ${weightDtype}; maxBuffer ` +
      `${((limits?.maxBufferSize ?? 0) / 1e9).toFixed(1)}GB, maxBinding ` +
      `${((limits?.maxStorageBufferBindingSize ?? 0) / 1e9).toFixed(1)}GB)`,
  });

  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  tokenizerLike = {
    encode: (text: string) => tokenizer.encode(text) as number[],
    decode: (ids: number[], options?: { skip_special_tokens?: boolean }) =>
      tokenizer.decode(ids, options) as string,
  };

  model = await loadGemma2FromUrl(
    api,
    `https://huggingface.co/${modelId}/resolve/main`,
    {
      maxSeqLen: 1024,
      weightDtype,
      onProgress: (loaded, total, status) => {
        const errs = getGpuUncapturedErrorCount();
        post({
          type: "progress",
          loaded,
          total,
          status:
            status + heapMB() + (errs > 0 ? ` · ⚠ ${errs} GPU errors` : ""),
        });
      },
      onTensorEvent: (ev) => post({ type: "tensor", ev }),
    },
  );

  sae = await loadSAE(saeBaseUrl, (status) =>
    post({ type: "progress", loaded: 0, total: 0, status: status + heapMB() }),
  );

  const errs = getGpuUncapturedErrorCount();
  if (errs > 0) {
    throw new Error(
      `Model+SAE loaded but ${errs} GPU validation errors occurred (see worker console).`,
    );
  }
  post({
    type: "loaded",
    modelId,
    weightDtype,
    numLayers: model.config.numLayers,
    hiddenSize: model.config.hiddenSize,
    saeLayer: sae.config.layer,
    numFeatures: sae.config.numFeatures,
    neuronpediaSaeId: sae.config.neuronpediaSaeId,
    tapeFlagSet: !!(globalThis as { __TORCHLETTE_ENV__?: unknown })
      .__TORCHLETTE_ENV__,
    tapeOn: STEP_TAPE_REPLAY,
  });
}

async function handleInspect(id: number, prompt: string, topK?: number) {
  if (!api || !model || !sae || !tokenizerLike) throw new Error("Not loaded");
  const report = await inspectFeatures(
    api,
    model,
    sae,
    tokenizerLike,
    prompt,
    topK ?? 20,
  );
  post({ type: "features", id, report });
}

async function handleGenerate(
  id: number,
  prompt: string,
  steer: SteerSpec[],
  maxNewTokens: number,
  temperature: number,
) {
  if (!api || !model || !sae || !tokenizerLike) throw new Error("Not loaded");
  const residualHook = makeSAEResidualHook(api, sae, steer);
  const stats = await generateChat(
    api,
    model,
    tokenizerLike,
    [{ role: "user", content: prompt }],
    {
      onDelta: (delta) => post({ type: "delta", id, delta }),
      onReplace: (text) => post({ type: "replace", id, text }),
    },
    {
      maxNewTokens,
      temperature,
      topK: 40,
      topP: 0.95,
      chat: false,
      residualHook,
    },
  );
  post({
    type: "done",
    id,
    stats: { ...stats, steered: residualHook !== undefined, steer },
  });
}

/** Debug (#59 gibberish hunt): read back raw embedding-table values and a
 *  gather result so the browser GPU table can be diffed against checkpoint
 *  ground truth. Driven via CDP: self.onmessage({data:{type:"readback"}}). */
async function handleReadback(row: number, n: number) {
  if (!api || !model) throw new Error("Not loaded");
  const w = model.embedTokens.weight;
  const table = await w.narrow(0, row, 1).narrow(1, 0, n).cpu();
  const idx = api.tensorFromArray([row], [1]);
  const gathered = await model.embedTokens
    .forward(idx)
    .narrow(1, 0, n)
    .cpu();
  const out = { row, dtype: w.dtype, table, gathered };
  (self as unknown as { __readback: unknown }).__readback = out;
  console.log("READBACK", JSON.stringify(out));
}

/** Synthetic f16-gather probe (no model needed): row r holds r + col/10, so a
 *  correct gather of row r reads [r.0, r.1, ...] and the Metal 2× bug reads
 *  [2r.0, ...]. Driven via CDP: self.onmessage({data:{type:"synthetic"}}). */
async function handleSynthetic() {
  if (!api) {
    const ok = await initWebGPU();
    if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
    api = new Torchlette("webgpu", { enableFusion: true });
  }
  const rows = 1000;
  const hidden = 8;
  const data = new Float32Array(rows * hidden);
  for (let r = 0; r < rows; r++)
    for (let col = 0; col < hidden; col++) data[r * hidden + col] = r + col / 10;
  const results: Record<string, number[]> = {};
  for (const dt of ["f16", "f32"] as const) {
    const table = api.tensorFromArray(data, [rows, hidden], {
      dtype: dt,
      device: "webgpu",
    });
    for (const r of [1, 5, 100]) {
      const idxT = api.tensorFromArray([r], [1]);
      results[`${dt}_row${r}`] = await api
        .embedding(table, idxT)
        .narrow(1, 0, 4)
        .cpu();
    }
  }
  (self as unknown as { __synthetic: unknown }).__synthetic = results;
  console.log("SYNTHETIC", JSON.stringify(results));
}

/** Decisive #59 isolation: dispatch LITERAL gather WGSL on a FRESH device,
 *  ZERO torchlette. Uploads a known f16 table [8,4] (row r col c = r + c/8,
 *  f16-exact) + f32 indices [1,4] at offset 0 and runs two literal shaders:
 *  (A) the old `array<f16>` gather, (B) the new `array<u32>` unpack2x16float
 *  gather. If A reads row 2r on Metal but B is correct → Tint→MSL miscompile.
 *  If A is correct on Metal too → the tab bug is NOT the shader; it's our
 *  dispatch params. Driven: self.onmessage({data:{type:"rawgather"}}). */
async function handleRawGather() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no adapter");
  const hasF16 = adapter.features.has("shader-f16");
  const device = await adapter.requestDevice({
    requiredFeatures: hasF16 ? (["shader-f16"] as GPUFeatureName[]) : [],
  });

  const N = 8;
  const D = 4;
  // Build the f16 table as raw bytes (row r col c = r + c/8, f16-exact).
  const f16bits = (x: number): number => {
    // minimal f32→f16 for small exact values
    const f = new Float32Array([x]);
    const u = new Uint32Array(f.buffer)[0];
    const sign = (u >>> 16) & 0x8000;
    let exp = ((u >>> 23) & 0xff) - 127 + 15;
    const mant = u & 0x7fffff;
    if (exp <= 0) return sign; // subnormal/zero (values here >=0 and small)
    return sign | (exp << 10) | (mant >>> 13);
  };
  const tableU16 = new Uint16Array(N * D);
  for (let r = 0; r < N; r++)
    for (let c = 0; c < D; c++) tableU16[r * D + c] = f16bits(r + c / 8);
  const indicesF32 = new Float32Array([1, 3, 5, 7]); // gather rows 1,3,5,7

  const mkBuf = (
    data: ArrayBufferView,
    usage: number,
  ): GPUBuffer => {
    const b = device.createBuffer({
      size: Math.max(16, (data.byteLength + 3) & ~3),
      usage,
      mappedAtCreation: true,
    });
    new Uint8Array(b.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    );
    b.unmap();
    return b;
  };

  const tableBuf = mkBuf(
    tableU16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const idxBuf = mkBuf(indicesF32, GPUBufferUsage.STORAGE);

  // out per shader: 4 f16 (one gathered row of D=4)
  const runShader = async (
    label: string,
    wgsl: string,
    outIsU16: boolean,
  ): Promise<number[] | string> => {
    try {
      const outBytes = outIsU16 ? D * 2 : D * 4;
      const outBuf = device.createBuffer({
        size: Math.max(16, (outBytes + 3) & ~3),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const cfg = new Uint32Array([D]); // size = D output elements
      const cfgBuf = mkBuf(cfg, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const mod = device.createShaderModule({ code: wgsl });
      const pipe = device.createComputePipeline({
        layout: "auto",
        compute: { module: mod, entryPoint: "main" },
      });
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tableBuf } },
          { binding: 1, resource: { buffer: idxBuf } },
          { binding: 2, resource: { buffer: outBuf } },
          { binding: 3, resource: { buffer: cfgBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1);
      pass.end();
      const readBuf = device.createBuffer({
        size: outBuf.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBuf.size);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const raw = readBuf.getMappedRange().slice(0);
      readBuf.unmap();
      // decode f16 out → numbers
      const u16 = new Uint16Array(raw);
      const decode = (h: number): number => {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7c00) >> 10;
        const m = h & 0x03ff;
        let val: number;
        if (e === 0) val = m / 1024 / 16384;
        else if (e === 31) val = m ? NaN : Infinity;
        else val = (1 + m / 1024) * Math.pow(2, e - 15);
        return s ? -val : val;
      };
      return Array.from({ length: D }, (_, i) => decode(u16[i]));
    } catch (err) {
      return `ERR: ${String(err)}`;
    }
  };

  // (A) OLD literal shader: array<f16> input, direct index. dim0 gather of [8,4].
  const shaderF16 = `enable f16;
struct Cfg { size: u32 };
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<uniform> config: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= config.size) { return; }
  let row = u32(indices[i]);
  let e = (row * ${D}u) + (i % ${D}u);
  out[i] = input[e];
}`;

  // (B) NEW literal shader: array<u32> input, unpack2x16float extraction.
  const shaderU32 = `enable f16;
struct Cfg { size: u32 };
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<uniform> config: Cfg;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= config.size) { return; }
  let row = u32(indices[i]);
  let e = (row * ${D}u) + (i % ${D}u);
  out[i] = f16(unpack2x16float(input[(e >> 1u)])[(e & 1u)]);
}`;

  const out: Record<string, unknown> = {
    hasF16,
    indices: Array.from(indicesF32),
    // expected: gather row 1 → [1, 1.125, 1.25, 1.375] (col/8), but the
    // dispatch gathers rows 1,3,5,7 at output cols 0..3 respectively:
    // out[i] = table[indices[i], i] = [1+0/8, 3+1/8, 5+2/8, 7+3/8]
    expected: [1 + 0 / 8, 3 + 1 / 8, 5 + 2 / 8, 7 + 3 / 8],
    A_f16_direct: await runShader("A", shaderF16, true),
    B_u32_unpack: await runShader("B", shaderU32, true),
    adapterInfo: {
      vendor: (adapter as unknown as { info?: { vendor?: string } }).info
        ?.vendor,
      architecture: (
        adapter as unknown as { info?: { architecture?: string } }
      ).info?.architecture,
    },
  };
  (self as unknown as { __rawgather: unknown }).__rawgather = out;
  console.log("RAWGATHER", JSON.stringify(out));
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await handleLoad(msg.modelId, msg.saeBaseUrl, msg.weightDtype);
    } else if (msg.type === "readback") {
      await handleReadback(msg.row ?? 0, msg.n ?? 16);
    } else if (msg.type === "synthetic") {
      await handleSynthetic();
    } else if (msg.type === "rawgather") {
      await handleRawGather();
    } else if (msg.type === "inspect") {
      await handleInspect(msg.id, msg.prompt, msg.topK);
    } else if (msg.type === "generate") {
      await handleGenerate(
        msg.id,
        msg.prompt,
        msg.steer ?? [],
        msg.maxNewTokens ?? 80,
        msg.temperature ?? 0.7,
      );
    }
  } catch (err) {
    post({ type: "error", id: msg.id, error: String(err) });
  }
};
