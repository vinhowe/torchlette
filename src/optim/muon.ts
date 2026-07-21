/**
 * Muon (Jordan et al. 2024 — "MomentUm Orthogonalized by Newton-schulz").
 *
 * The CONTRACTION dividend of the Crystal-3 tail (docs/semantic-derivation-design.md
 * §2 category d, §16 deferral CASHED): a complete optimizer realized from a
 * definition — `MUON_PROGRAM` (src/ops/semantic/optimizer.ts) — exactly as Lion
 * was, but reaching ABOVE the pure-elementwise algebra into the ONE contraction
 * node (`mm`) the frame previously lacked. NO hand kernel and NO hand grads: the
 * Newton-Schulz orthogonalization is an UNROLLED composition of matmul
 * contractions that the program interpreter realizes over `rt.matmul` (the
 * existing kernel, referenced not re-owned — design §4.4).
 *
 *   buf' = μ·buf + g                                     (momentum, SGD-form state)
 *   O    = NS(buf)   ≈ orthogonalize(buf)   (Newton-Schulz; see MUON_ORTHO)
 *   p'   = p − lr·( √max(1,rows/cols)·O + wd·p )         (decoupled weight decay)
 *
 * This realizer sequences the DECLARED EFFECTS + POLICY (design §4.6): the
 * in-place momentum copy_, the step boundary, the ORIENTATION transpose (Newton-
 * Schulz needs rows≤cols; the momentum is transposed and the result transposed
 * back for tall matrices), and the AdamW ROUTING of embedding/1D params (standard
 * Muon practice — orthogonalization is only meaningful for 2D weight matrices).
 * The arithmetic DERIVES from the program; the routing/effect is declared.
 */

import type { DeviceKind } from "../backend/types";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import {
  buildMuonOrtho,
  evalOptTensor,
  MUON_M_NEW,
  MUON_NS_STEPS,
  MUON_P_APPLY,
  MUON_PROGRAM,
  type OptRoles,
  type OptTerm,
} from "../ops/semantic/optimizer";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { Adam } from "./adam";
import { assertFlattenable, OptimizerPackRefusal } from "./pack-optimizer";
import { validateOptimizerParams } from "./validate";

export type MuonOptions = {
  /** Learning rate for the Muon-routed (2D weight) params. */
  lr: number;
  /** Momentum coefficient μ for the buffer (default 0.95, standard Muon). */
  momentum?: number;
  /** Decoupled weight decay for Muon params (default 0). */
  weightDecay?: number;
  /** Newton-Schulz iteration count (default 5, the published quintic budget). */
  nsSteps?: number;
  /**
   * Params to route to the internal AdamW instead of Muon — the embeddings /
   * lm_head (standard Muon practice). 1D params (biases, norms) are auto-routed
   * to AdamW regardless. Everything else 2D is orthogonalized by Muon.
   */
  adamwParams?: Tensor[];
  /** AdamW fallback hypers (for the excluded / 1D params). */
  adamwLr?: number;
  adamwBetas?: [number, number];
  adamwEps?: number;
  adamwWeightDecay?: number;
};

/** Frobenius-norm guard: 1/sqrt(‖·‖²+ε²) ≈ 1/(‖·‖+ε) away from zero, no scalar div. */
const NS_EPS2 = 1e-14;

export class Muon {
  private readonly params: Tensor[];
  private readonly api: Torchlette;
  private readonly device: DeviceKind;
  private readonly momentumCoeff: number;
  private readonly lr: number;
  private readonly weightDecay: number;
  private readonly nsSteps: number;
  /** The unrolled Newton-Schulz orthogonalization term (built once per nsSteps). */
  private readonly orthoTerm: OptTerm;
  /** The 2D params Muon orthogonalizes (index → param); others go to AdamW. */
  private readonly muonParams: Tensor[];
  /** Per-Muon-param momentum buffer `buf`. Lazily created (persistent state). */
  private readonly buffers: Array<RuntimeTensor | null>;
  /** The internal AdamW for embedding/1D/excluded params (null if none). */
  private readonly adamw: Adam | null;
  private _lr: number;
  /** Cached once-per-class pack verdict (chain-packing P3, design §6.1). */
  private _packRefusal: OptimizerPackRefusal | null = null;
  private _packVerdictDecided = false;

  constructor(params: Tensor[], options: MuonOptions, api?: Torchlette) {
    const { api: engine, device } = validateOptimizerParams(
      "Muon",
      params,
      api,
    );
    if (options.lr <= 0) throw new Error("Muon learning rate must be > 0");
    const momentum = options.momentum ?? 0.95;
    if (momentum < 0 || momentum >= 1)
      throw new Error("Muon momentum must be in [0, 1)");
    const nsSteps = options.nsSteps ?? MUON_NS_STEPS;
    if (nsSteps < 1) throw new Error("Muon nsSteps must be ≥ 1");

    this.api = engine;
    this.device = device;
    this.params = params;
    this.momentumCoeff = momentum;
    this.lr = options.lr;
    this._lr = options.lr;
    this.weightDecay = options.weightDecay ?? 0;
    this.nsSteps = nsSteps;
    this.orthoTerm = buildMuonOrtho(nsSteps);

    // ROUTING (policy as data): Muon orthogonalizes 2D weight matrices that are
    // NOT explicitly excluded; embeddings/lm_head + all 1D params fall back to
    // AdamW (standard Muon practice — NS orthogonalization needs a matrix).
    const excluded = new Set<Tensor>(options.adamwParams ?? []);
    this.muonParams = [];
    const adamwParams: Tensor[] = [];
    for (const p of params) {
      if (p.shape.length === 2 && !excluded.has(p)) this.muonParams.push(p);
      else adamwParams.push(p);
    }
    this.buffers = new Array(this.muonParams.length).fill(null);

    this.adamw =
      adamwParams.length > 0
        ? new Adam(
            adamwParams,
            {
              lr: options.adamwLr ?? options.lr,
              betas: options.adamwBetas ?? [0.9, 0.95],
              eps: options.adamwEps ?? 1e-8,
              weightDecay: options.adamwWeightDecay ?? this.weightDecay,
              adamW: true,
            },
            engine,
          )
        : null;
  }

  getParams(): Tensor[] {
    return this.params.slice();
  }

  getLR(): number {
    return this._lr;
  }

  setLR(lr: number): void {
    this._lr = lr;
  }

  /** How many params Muon orthogonalizes (vs the AdamW fallback). */
  get numMuonParams(): number {
    return this.muonParams.length;
  }

  /**
   * The Muon class's pack verdict, declared to the packer ONCE and cached
   * (chain-packing P3 — design §6.1, "Muon full-refusal v1"). Muon declares
   * `MUON_PROGRAM` to the packer exactly as Adam/Lion/SGD do; because the
   * program carries an `mm` node (Newton–Schulz orthogonalization), the packer's
   * clause-4 hard gate (`assertFlattenable`) refuses with a typed, named
   * `OptimizerPackRefusal`, so Muon realizes its step through the per-param path.
   *
   * The verdict is a function of the PROGRAM alone (never the live shapes), so it
   * is decided ONCE and cached — re-raising every step would be churn (design
   * §6.1: "cheap, once-per-class decision, not per-step re-throw"). Returns the
   * refusal (typed/named) for introspection, or `null` if the class were
   * flat-packable — unreachable in v1, the seam a future elementwise partial pack
   * (§6.1) would consume.
   */
  packVerdict(): OptimizerPackRefusal | null {
    if (!this._packVerdictDecided) {
      this._packVerdictDecided = true;
      try {
        assertFlattenable(MUON_PROGRAM, MUON_PROGRAM.paramUpdate);
      } catch (e) {
        if (!(e instanceof OptimizerPackRefusal)) throw e;
        this._packRefusal = e;
      }
    }
    return this._packRefusal;
  }

  step(): Tensor[] {
    const rt = this.api._runtime();

    // chain-packing P3 (design §6.1 — Muon full-refusal v1): declare the class
    // to the packer. `MUON_PROGRAM` carries `mm` (Newton–Schulz), so the packer
    // refuses (typed `OptimizerPackRefusal`, cached once-per-class) and the step
    // realizes each 2D param through the per-param path below — byte-identical to
    // the pre-routing math. A future elementwise partial pack (§6.1) would emit
    // its accepted segments on a non-refusal verdict.
    this.packVerdict();

    for (let i = 0; i < this.muonParams.length; i++) {
      const param = this.muonParams[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) continue;

      // Lazy persistent momentum buffer. registerState adopts it into the step
      // snapshot (else markStep demotes the mid-step-created tensor and pools its
      // buffer while live — the silent-UAF class Adam/Lion also guard).
      let buf = this.buffers[i];
      if (!buf) {
        buf = rt.registerState(rt.zeros(param.shape, this.device));
        this.buffers[i] = buf;
      }

      // Every intermediate this param CREATES (the momentum arithmetic, the NS
      // contraction chain, the transposes/scale, the apply) is collected and
      // DISPOSED after the copy_s are built — Adam foreach's discipline. Without
      // it, each full-model-size NS intermediate is conservatively held as
      // "user-owned" for the whole step and the executor's liveness cannot
      // release it (32GB blows on the NS chain). Disposal removes the wrapper
      // from the pending registry; the IR nodes survive (nodes reference nodes)
      // so the copy_s still execute correctly.
      const sink: RuntimeTensor[] = [];
      const track = (t: RuntimeTensor): RuntimeTensor => {
        sink.push(t);
        return t;
      };

      // buf' = μ·buf + g, IN PLACE. Read the POST-copy_ buffer ref for the
      // orthogonalization so the param chain depends on the copy (Adam's
      // dangling-copy_ discipline — reading mNew directly strands the copy).
      const mNew = evalOptTensor(
        MUON_M_NEW,
        rt,
        { m: buf, g: grad, mu: this.momentumCoeff } as OptRoles,
        sink,
      );
      rt.copy_(buf, mNew);

      // ORIENTATION policy: Newton-Schulz's X·Xᵀ is rank-limited to min(rows,cols),
      // so run it with rows ≤ cols and transpose the result back for tall params.
      const [R, C] = param.shape;
      const tall = R > C;
      const oriented = tall
        ? track(rt.transpose(buf, { dim0: 0, dim1: 1 }))
        : buf;

      // ns_scale = 1/√(‖buf‖² + ε²) — normalizes into the NS convergence basin
      // (Frobenius-norm invariant to the orientation transpose). Stays on-device.
      const sumSq = track(rt.sum(track(rt.mul(oriented, oriented))));
      const nsScale = track(rt.rsqrt(track(rt.add(sumSq, NS_EPS2))));

      // The orthogonalization DERIVES from the program (unrolled NS contractions).
      const orthO = evalOptTensor(
        this.orthoTerm,
        rt,
        { m: oriented, ns_scale: nsScale } as OptRoles,
        sink,
      );
      const backOriented = tall
        ? track(rt.transpose(orthO, { dim0: 0, dim1: 1 }))
        : orthO;

      // rms magnitude match (√max(1,rows/cols)) — a shape scalar (POLICY), then
      // p' = p − lr·(u + wd·p) via MUON_P_APPLY (the derived apply tail).
      const rms = Math.sqrt(Math.max(1, R / C));
      const u = track(rt.mul(backOriented, rms));
      const pNew =
        this.weightDecay !== 0
          ? evalOptTensor(
              MUON_P_APPLY,
              rt,
              {
                p: param._unwrap(),
                u,
                lr: this._lr,
                wd: this.weightDecay,
              } as OptRoles,
              sink,
            )
          : track(rt.sub(param._unwrap(), track(rt.mul(u, this._lr))));
      rt.copy_(param._unwrap(), pNew);

      // Release this param's derived intermediates (buf/param are persistent and
      // never entered the sink — they are role inputs, not created here).
      for (const t of new Set(sink))
        (t as { dispose?: () => void }).dispose?.();
    }

    // The excluded / 1D params take an AdamW step (its own queued boundary).
    if (this.adamw) this.adamw.step();

    // Implied step boundary (minimal training loops): commits at the next
    // backward() or explicit markStep().
    this.api.queueStepBoundary();
    return this.params;
  }

  zeroGrad(): void {
    for (const param of this.params) param.zeroGrad();
  }
}
