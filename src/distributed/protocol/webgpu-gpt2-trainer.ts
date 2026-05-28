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
import { Adam } from "../../optim/index.ts";
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
   * Implementations may cache, prefetch, retry, etc.
   */
  fetch(minTokens: number): Promise<number[]>;
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
  loraRank?: number;
  loraAlpha?: number;
  fullFinetuning?: boolean;
  checkpointing?: boolean;
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
  loraRank: 1,
  loraAlpha: 1,
  fullFinetuning: true,
  checkpointing: true,
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

  /** Anchor params snapshot on CPU. Updated by setAnchor / applyOuterStep. */
  private anchor: Float32Array[] = [];

  private initialized = false;
  private tokensCache: number[] | null = null;

  constructor(opts: WebGPUGPT2TrainerOptions) {
    this.api = opts.api;
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
      loraRank: merged.loraRank,
      loraAlpha: merged.loraAlpha,
      fullFinetuning: merged.fullFinetuning,
      checkpointing: merged.checkpointing,
    };
  }

  /**
   * Build the model, init params, create optimizers. Must be called once
   * before any Trainer interface method.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    const { GPT2WithLoRA } = await import(
      "../../../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora.ts"
    );
    this.model = new GPT2WithLoRA(
      this.api,
      this.opts.modelConfig,
      { rank: this.opts.loraRank, alpha: this.opts.loraAlpha },
      "webgpu",
    );
    this.params = this.model.getAllParameters();

    // Seed-init via PyTorch-style normal_ for matrix-rank params.
    for (const p of this.params) {
      if (p.shape.length >= 2) normal_(this.api, p, 0, 0.02);
    }
    await this.api._runtime().forceAllPending();
    this.opts.log(
      `Initialized GPT-2 (${this.params.length} param tensors, ${this.totalParamCount().toLocaleString()} elements)`,
    );

    this.model.train(true);
    this.model.enableCheckpointing(this.opts.checkpointing);
    if (this.opts.fullFinetuning) {
      this.model.fullCheckpoint = true;
      this.model.setFullFinetuning(true);
    }

    this.innerOpt = new Adam(
      this.params,
      { lr: this.opts.innerLr, weightDecay: this.opts.weightDecay },
      this.api,
    );
    this.outerOpt = new NesterovOuterOptimizer(this.api, {
      lr: this.opts.outerLr,
      momentum: this.opts.outerMu,
    });
    this.accumGrads = this.params.map((p) =>
      this.api.zeros(p.shape, { device: "webgpu" }),
    );
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
      const offset = (round * this.opts.innerSteps + step) * tokensPerStep;
      totalLoss += await this.singleInnerStep(tokens, offset);
    }
    return totalLoss / this.opts.innerSteps;
  }

  private async singleInnerStep(
    tokens: number[],
    offset: number,
  ): Promise<number> {
    const api = this.api;
    const opts = this.opts;
    const maxStart = Math.max(1, tokens.length - opts.seqLen - 1);
    let totalLoss = 0;

    for (const ag of this.accumGrads) api.zero_(ag);

    for (let acc = 0; acc < opts.accumSteps; acc++) {
      const microOffset = offset + acc * opts.batchSize * opts.seqLen;
      const inputData: number[] = [];
      const targetData: number[] = [];
      for (let b = 0; b < opts.batchSize; b++) {
        const start = (microOffset + b * opts.seqLen) % maxStart;
        for (let i = 0; i < opts.seqLen; i++) {
          inputData.push(tokens[start + i]);
          targetData.push(tokens[start + i + 1]);
        }
      }
      await api.beginStep();
      const input = api.tensorFromArray(
        inputData,
        [opts.batchSize, opts.seqLen],
        { device: "webgpu" },
      );
      const target = api.tensorFromArray(
        targetData,
        [opts.batchSize, opts.seqLen],
        { device: "webgpu" },
      );
      const loss = api.tidy(() => {
        const l = api.autocast(
          () => this.model.forwardWithLoss(input, target).loss,
        );
        api.keep(l);
        return l;
      });
      totalLoss += await loss.item();
      await loss.backward();
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
    clipGradNorm_(api, this.params, 1.0);
    this.innerOpt.step();
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
    await this.outerOpt.stepFromCpu(this.params, this.anchor, avgGrad);
    // Anchor = new params after outer step.
    const anchor: Float32Array[] = [];
    for (const p of this.params) {
      anchor.push(new Float32Array(await p.cpu()));
    }
    this.anchor = anchor;
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
