/**
 * WebGPU GPT-2 Trainer adapter.
 *
 * Implements the protocol's Trainer interface against a real WebGPU
 * GPT-2 model + Adam (inner) + Nesterov (outer) optimizer. Lifts the
 * training logic out of the monolithic diloco-webrtc-agent so the
 * barrier state machines can drive it the same way they drive the
 * StubTrainer in tests.
 *
 * Token data is abstracted behind TokenSource — production wires the
 * HuggingFace dataset; tests can supply a stub.
 */

import { clipGradNorm_ } from "../../nn/index.ts";
import { normal_ } from "../../nn/init.ts";
import { Adam, GradScaler } from "../../optim/index.ts";
import type { Tensor } from "../../frontend/tensor.ts";
import type { Torchlette } from "../../frontend/torchlette.ts";
import { NesterovOuterOptimizer } from "../outer-optimizer.ts";
import type { ParamShapes, Trainer } from "./trainer.ts";

export interface GPT2ModelConfig {
  vocabSize: number;
  blockSize: number;
  numLayers: number;
  numHeads: number;
  embedDim: number;
  dropoutRate: number;
}

export interface TokenSource {
  /**
   * Fetch at least `minTokens` tokens for the next round's inner training.
   * Implementations may cache, prefetch, retry, etc. Returning ArrayLike
   * (not just number[]) lets callers pass a Uint16Array directly without
   * a number[] copy — important when the cache is hundreds of MB.
   */
  fetch(minTokens: number): Promise<ArrayLike<number>>;
}

export interface WebGPUGPT2TrainerOptions {
  api: Torchlette;
  modelConfig: GPT2ModelConfig;
  tokenSource: TokenSource;
  innerLr?: number;
  outerLr?: number;
  outerMu?: number;
  innerSteps?: number;
  batchSize?: number;
  seqLen?: number;
  accumSteps?: number;
  weightDecay?: number;
  /** Gradient checkpointing during forward; trades compute for memory. */
  checkpointing?: boolean;
  /** Wrap forward in api.autocast (fp16 inputs to matmul). Defaults true. */
  useAutocast?: boolean;
  /** Clip gradient norm to this value before optimizer.step. 0 disables. */
  gradClipNorm?: number;
  /**
   * Override window-start sampling. Given the linear inner-step index and
   * batch size, returns `batchSize` start offsets into the token cache.
   * Defaults to a per-step LCG. Used by the cross-framework parity harness
   * to feed bit-identical windows to torchlette and the PyTorch baseline.
   */
  sampleWindowStarts?: (
    stepIndex: number,
    batchSize: number,
    maxStart: number,
  ) => number[];
  /** Logger; defaults to console.error. */
  log?: (msg: string) => void;
}

const defaultOpts = {
  innerLr: 1e-4,
  outerLr: 0.7,
  outerMu: 0.9,
  innerSteps: 20,
  batchSize: 4,
  seqLen: 512,
  accumSteps: 1,
  weightDecay: 0.1,
  checkpointing: true,
  useAutocast: true,
  gradClipNorm: 1.0,
} as const;

export class WebGPUGPT2Trainer implements Trainer {
  private readonly api: Torchlette;
  private readonly opts: Required<
    Omit<WebGPUGPT2TrainerOptions, "api" | "modelConfig" | "tokenSource" | "log">
  > & {
    api: Torchlette;
    modelConfig: GPT2ModelConfig;
    tokenSource: TokenSource;
    log: (msg: string) => void;
  };

  // biome-ignore lint/suspicious/noExplicitAny: GPT2WithLoRA shape isn't typed
  private model!: any;
  private params: Tensor[] = [];
  private innerOpt!: Adam;
  private outerOpt!: NesterovOuterOptimizer;
  private accumGrads: Tensor[] = [];
  private scaler: GradScaler | null = null;

  /** Anchor params snapshot on CPU. Updated by setAnchor / applyOuterStep. */
  private anchor: Float32Array[] = [];

  private initialized = false;
  private tokensCache: ArrayLike<number> | null = null;
  private readonly sampleWindowStarts?: (
    stepIndex: number,
    batchSize: number,
    maxStart: number,
  ) => number[];

  constructor(opts: WebGPUGPT2TrainerOptions) {
    this.api = opts.api;
    this.sampleWindowStarts = opts.sampleWindowStarts;
    const merged = { ...defaultOpts, ...opts };
    this.opts = {
      api: opts.api,
      modelConfig: opts.modelConfig,
      tokenSource: opts.tokenSource,
      log: opts.log ?? ((m) => console.error(`[trainer] ${m}`)),
      innerLr: merged.innerLr,
      outerLr: merged.outerLr,
      outerMu: merged.outerMu,
      innerSteps: merged.innerSteps,
      batchSize: merged.batchSize,
      seqLen: merged.seqLen,
      accumSteps: merged.accumSteps,
      weightDecay: merged.weightDecay,
      checkpointing: merged.checkpointing,
      useAutocast: merged.useAutocast,
      gradClipNorm: merged.gradClipNorm,
    };
  }

  /**
   * Build the model, init params, create optimizers. Must be called once
   * before any Trainer interface method.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    // Use the plain (non-LoRA) GPT-2 model. The trainer drives full
    // fine-tuning from random init, so the LoRA-wrapped version would only
    // add a structurally dead branch (loraA/loraB stay at zero through
    // training because the gradient chain through a zero-init outer matrix
    // contributes zero back). Plain GPT2 has the smaller param list, no
    // setFullFinetuning toggles, and passes checkpointing through forward
    // options instead of via mutable state.
    const { GPT2 } = await import(
      "../../../examples/gpt2/model.ts"
    );
    this.model = new GPT2(this.api, this.opts.modelConfig, {
      device: "webgpu",
    });
    this.params = this.model.parameters();

    // nanoGPT / GPT-2 paper init (must match the PyTorch baseline exactly):
    //   - Linear/Embedding weight (2D ".weight"): N(0, 0.02)
    //   - residual output projections (".cProj.weight"): N(0, 0.02/sqrt(2L))
    //     — scaled down so residual-stream variance stays ~constant with
    //     depth (GPT-2 §2.3). Both attention and MLP output projections.
    //   - bias (".bias"): 0
    //   - LayerNorm weight (1D ".weight"): leave at nn default (1.0)
    const numLayers = this.opts.modelConfig.numLayers;
    const residStd = 0.02 / Math.sqrt(2 * numLayers);
    for (const [name, p] of this.model.namedParameters()) {
      if (name.endsWith(".bias")) {
        this.api.zero_(p);
      } else if (name.endsWith("cProj.weight")) {
        normal_(this.api, p, 0, residStd);
      } else if (p.shape.length >= 2) {
        normal_(this.api, p, 0, 0.02);
      }
      // 1D ".weight" (LayerNorm scale) stays at its nn default of 1.0.
    }
    await this.api._runtime().forceAllPending();
    this.opts.log(
      `Initialized GPT-2 (${this.params.length} param tensors, ${this.totalParamCount().toLocaleString()} elements)`,
    );

    this.model.train(true);

    this.innerOpt = new Adam(
      this.params,
      {
        lr: this.opts.innerLr,
        weightDecay: this.opts.weightDecay,
        adamW: true,
      },
      this.api,
    );
    this.outerOpt = new NesterovOuterOptimizer(this.api, {
      lr: this.opts.outerLr,
      momentum: this.opts.outerMu,
    });
    this.accumGrads = this.params.map((p) =>
      this.api.zeros(p.shape, { device: "webgpu" }),
    );
    if (this.opts.useAutocast) {
      // GradScaler is required when forward runs in fp16 — without it,
      // small-magnitude gradients underflow during the bf16/fp16 backward
      // and training stalls (loss plateaus high). 1024 is the torchlette
      // example default; the scaler auto-adjusts via inf detection.
      this.scaler = new GradScaler(this.api, { initScale: 1024.0 });
    }
    await this.api._runtime().forceAllPending();

    this.initialized = true;

    // Populate the anchor from the initialized params so snapshotAnchor()
    // never returns empty. The state machine will call setAnchor() again
    // before training starts, but other peers may request F16W in the gap
    // between transport registration and run() — they need real params,
    // not an empty array. Setting it here makes the trainer's anchor
    // invariant "always reflects some valid params" hold post-init.
    const anchor: Float32Array[] = [];
    for (const p of this.params) {
      anchor.push(new Float32Array(await p.cpu()));
    }
    this.anchor = anchor;
  }

  paramShapes(): ParamShapes {
    return this.params.map((p) => p.shape.slice());
  }

  totalParamCount(): number {
    let total = 0;
    for (const p of this.params) {
      total += p.shape.reduce((a, b) => a * b, 1);
    }
    return total;
  }

  async setAnchor(): Promise<void> {
    this.requireInit();
    const anchor: Float32Array[] = [];
    for (const p of this.params) {
      anchor.push(new Float32Array(await p.cpu()));
    }
    this.anchor = anchor;
  }

  async innerSteps(round: number): Promise<number> {
    this.requireInit();
    // Pull fresh tokens — at least enough to cover one round of training.
    const tokensPerStep =
      this.opts.batchSize * this.opts.accumSteps * this.opts.seqLen;
    const targetTokens = this.opts.innerSteps * tokensPerStep;
    this.tokensCache = await this.opts.tokenSource.fetch(targetTokens);
    const tokens = this.tokensCache;

    let totalLoss = 0;
    for (let step = 0; step < this.opts.innerSteps; step++) {
      const stepIndex = round * this.opts.innerSteps + step;
      const offset = stepIndex * tokensPerStep;
      totalLoss += await this.singleInnerStep(tokens, offset, stepIndex);
    }
    return totalLoss / this.opts.innerSteps;
  }

  private async singleInnerStep(
    tokens: ArrayLike<number>,
    offset: number,
    stepIndex: number,
  ): Promise<number> {
    const api = this.api;
    const opts = this.opts;
    const scaler = this.scaler;
    const maxStart = Math.max(1, tokens.length - opts.seqLen - 1);
    let totalLoss = 0;

    if (scaler) {
      // Reads back the inf-detection result from the previous step (no-op on
      // the first step). Must run before scale() so the new scale factor is
      // active for this step.
      await scaler.resolveDeferred();
    }

    // Build per-batch random window starts for every microbatch up front.
    // The LCG is seeded per-step so two peers with the same SEED + cache
    // see the same windows — important for the regression harness.
    const microOffsetBase = offset;
    const innerStepBatches: { input: number[]; target: number[] }[] = [];
    for (let acc = 0; acc < opts.accumSteps; acc++) {
      const microOffset = microOffsetBase + acc * opts.batchSize * opts.seqLen;
      const inputData: number[] = [];
      const targetData: number[] = [];
      // Window starts come from the injected sampler if present (parity
      // harness feeds shared offsets), else a per-step LCG.
      let starts: number[];
      if (this.sampleWindowStarts) {
        const microStepIndex = stepIndex * opts.accumSteps + acc;
        starts = this.sampleWindowStarts(microStepIndex, opts.batchSize, maxStart);
      } else {
        starts = [];
        let rng = (microOffset * 2654435761) >>> 0;
        for (let b = 0; b < opts.batchSize; b++) {
          rng = ((rng * 1103515245 + 12345) & 0x7fffffff) >>> 0;
          starts.push(rng % maxStart);
        }
      }
      for (let b = 0; b < opts.batchSize; b++) {
        const start = starts[b]!;
        for (let i = 0; i < opts.seqLen; i++) {
          inputData.push(tokens[start + i]);
          targetData.push(tokens[start + i + 1]);
        }
      }
      innerStepBatches.push({ input: inputData, target: targetData });
    }

    // accumSteps=1: single forward+backward+step inside one beginStep,
    // matching the standalone regression harness exactly. Skipping the
    // accumGrads dance is both an FLOP saving and a precision win — the
    // extra mul(accumGrads, 1) lazy op introduces an additional materialization
    // boundary that produced a measurable ~0.7-0.8 nat convergence gap vs.
    // the no-accum path.
    if (opts.accumSteps === 1) {
      const batch = innerStepBatches[0]!;
      await api.beginStep();
      const input = api.tensorFromArray(
        batch.input,
        [opts.batchSize, opts.seqLen],
        { device: "webgpu" },
      );
      const target = api.tensorFromArray(
        batch.target,
        [opts.batchSize, opts.seqLen],
        { device: "webgpu" },
      );
      const loss = api.tidy(() => {
        const fwd = () =>
          this.model.forwardWithLoss(input, target, {
            useCheckpoint: this.opts.checkpointing,
          }).loss;
        const l = this.opts.useAutocast ? api.autocast(fwd) : fwd();
        api.keep(l);
        return l;
      });
      totalLoss += await loss.item();
      const backwardTarget = scaler ? scaler.scale(loss) : loss;
      await backwardTarget.backward();
      if (scaler) scaler.unscale_(this.innerOpt);
      if (this.opts.gradClipNorm > 0) {
        clipGradNorm_(api, this.params, this.opts.gradClipNorm);
      }
      if (scaler) {
        scaler.step(this.innerOpt);
        scaler.update();
      } else {
        this.innerOpt.step();
      }
      this.innerOpt.zeroGrad();
      input.dispose();
      target.dispose();
      api.endStep();
      await api.markStep();
      return totalLoss / opts.accumSteps;
    }

    // accumSteps>1: accumulate gradients across micro-batches, then step.
    for (const ag of this.accumGrads) api.zero_(ag);

    for (let acc = 0; acc < opts.accumSteps; acc++) {
      const batch = innerStepBatches[acc]!;
      await api.beginStep();
      const input = api.tensorFromArray(
        batch.input,
        [opts.batchSize, opts.seqLen],
        { device: "webgpu" },
      );
      const target = api.tensorFromArray(
        batch.target,
        [opts.batchSize, opts.seqLen],
        { device: "webgpu" },
      );
      const loss = api.tidy(() => {
        const fwd = () =>
          this.model.forwardWithLoss(input, target, {
            useCheckpoint: this.opts.checkpointing,
          }).loss;
        const l = this.opts.useAutocast ? api.autocast(fwd) : fwd();
        api.keep(l);
        return l;
      });
      totalLoss += await loss.item();
      const backwardTarget = scaler ? scaler.scale(loss) : loss;
      await backwardTarget.backward();
      for (let i = 0; i < this.params.length; i++) {
        // biome-ignore lint/suspicious/noExplicitAny: tensor.grad isn't strongly typed
        const grad = (this.params[i] as any).grad;
        if (grad) {
          api.add_(this.accumGrads[i], api.mul(grad, 1 / opts.accumSteps));
        }
      }
      await api._runtime().forceAllPending();
      this.innerOpt.zeroGrad();
      input.dispose();
      target.dispose();
      api.endStep();
      await api.markStep();
    }

    await api.beginStep();
    for (let i = 0; i < this.params.length; i++) {
      // biome-ignore lint/suspicious/noExplicitAny: tensor._setGrad isn't strongly typed
      (this.params[i] as any)._setGrad(api.mul(this.accumGrads[i], 1));
    }
    if (scaler) {
      scaler.unscale_(this.innerOpt);
      if (this.opts.gradClipNorm > 0) {
        clipGradNorm_(
          api,
          this.params,
          this.opts.gradClipNorm * scaler.getScale(),
        );
      }
      scaler.step(this.innerOpt);
      scaler.update();
    } else {
      if (this.opts.gradClipNorm > 0) {
        clipGradNorm_(api, this.params, this.opts.gradClipNorm);
      }
      this.innerOpt.step();
    }
    this.innerOpt.zeroGrad();
    api.endStep();
    await api.markStep();

    return totalLoss / opts.accumSteps;
  }

  async pseudograd(): Promise<Float32Array[]> {
    this.requireInit();
    const out: Float32Array[] = [];
    for (let i = 0; i < this.params.length; i++) {
      const cur = await this.params[i].cpu();
      const anc = this.anchor[i];
      const delta = new Float32Array(cur.length);
      for (let j = 0; j < cur.length; j++) delta[j] = cur[j] - anc[j];
      out.push(delta);
    }
    return out;
  }

  async applyOuterStep(avgGrad: Float32Array[]): Promise<void> {
    this.requireInit();
    if (avgGrad.length !== this.params.length) {
      throw new Error(
        `applyOuterStep: payload tensor count ${avgGrad.length} != ${this.params.length}`,
      );
    }
    // The outer step returns the exact f32 values it uploaded into the GPU
    // params — adopt them as the new anchor directly. Reading the model back
    // here (the previous code) cost a full-model GPU->CPU transfer per round
    // to fetch bytes we already hold.
    this.anchor = await this.outerOpt.stepFromCpu(
      this.params,
      this.anchor,
      avgGrad,
    );
  }

  async revertToAnchor(): Promise<void> {
    this.requireInit();
    await this.api.beginStep();
    for (let i = 0; i < this.params.length; i++) {
      this.api.copy_(
        this.params[i],
        this.api.tensorFromArray(this.anchor[i], this.params[i].shape, {
          device: "webgpu",
        }),
      );
    }
    this.api.endStep();
    await this.api.markStep();
  }

  async snapshotAnchor(): Promise<Float32Array[]> {
    this.requireInit();
    return this.anchor.map((a) => new Float32Array(a));
  }

  async applyF16W(params: Float32Array[]): Promise<void> {
    this.requireInit();
    if (params.length !== this.params.length) {
      throw new Error(
        `applyF16W: payload tensor count ${params.length} != ${this.params.length}`,
      );
    }
    await this.api.beginStep();
    for (let i = 0; i < this.params.length; i++) {
      this.api.copy_(
        this.params[i],
        this.api.tensorFromArray(params[i], this.params[i].shape, {
          device: "webgpu",
        }),
      );
    }
    this.api.endStep();
    await this.api.markStep();
    this.anchor = params.map((p) => new Float32Array(p));
  }

  async resetOptimState(): Promise<void> {
    this.requireInit();
    this.innerOpt.resetState();
    this.outerOpt.reset();
  }

  private requireInit(): void {
    if (!this.initialized) {
      throw new Error("WebGPUGPT2Trainer: call initialize() before use");
    }
  }
}
