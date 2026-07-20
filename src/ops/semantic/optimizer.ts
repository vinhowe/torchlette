/**
 * Semantic Derivation — OPTIMIZERS AS PROGRAMS (Crystal Campaign 3, Phase 5).
 *
 * An optimizer's update is a SEMANTIC COMPOSITION: a small dataflow of the
 * primitive algebra over the named state/hyperparameter roles it maintains
 * (`m`, `v`, `g`, `p`, `lr`, `bc1`, `bc2`, `eps`, `wd`, `beta*`). This is the
 * design §4.6 boundary made concrete: an optimizer's ARITHMETIC is a derivable
 * term (an `OptTerm` over roles), while its in-place m/v state + step boundary
 * are DECLARED EFFECTS owned by the realizer (adam.ts / lion.ts) — the
 * arithmetic derives, the effect is declared.
 *
 * The fused adamStep kernel is the composition's DERIVED REALIZATION (schedule-
 * state P4 precedent: the kernel altitude is already done). This module is the
 * GRAPH altitude: the foreach + elementwise paths GENERATE their moment/param
 * chains by interpreting a program (`evalOptTerm`), so the hand-written
 * `add(mul(m,β1), …)` chains — copied across the foreach and elementwise paths —
 * stop existing. The fused kernel is byte-ASSERTED against the same program
 * (test/semantic-optimizer.spec.ts), not re-owned (design §4.4, RT3 seam).
 *
 * A program is DATA — no leaf is a JS closure / WGSL string / buffer — proven by
 * `assertNoOptimizerProgramBody` (the P0 `assertNoDefinitionBody` analogue for
 * the optimizer frame; the covenant/R22 defense that keeps the byte differential
 * un-gameable).
 *
 * SGD (+momentum), LION, and MUON are SIBLING definitions (the generality
 * proof): Lion is sign-based, ~free from the elementwise algebra; MUON adds the
 * ONE contraction node (`mm`) the frame lacked and orthogonalizes via unrolled
 * Newton–Schulz — an optimizer from its definition, no hand kernel, no hand
 * grads (optimizers never need a VJP; §4.6). The `mm` node emits the existing
 * `rt.matmul` (referenced, not re-owned — §4.4, category d).
 */

import type { RuntimeEngine } from "../../runtime/engine";
import type { Tensor as RuntimeTensor } from "../../runtime/tensor";

// ----------------------------------------------------------------------------
// The OptTerm schema — a term over NAMED roles (the design §4.6 "update: Expr").
// Pure elementwise sub-algebra: the roles carry the state/hypers, the primitives
// are the arithmetic optimizers use (add/sub/mul/div + sqrt/sign/abs/neg/exp).
// ----------------------------------------------------------------------------

export type OptUnary = "neg" | "sqrt" | "sign" | "abs" | "exp";
export type OptBinary = "add" | "sub" | "mul" | "div";

export type OptTerm =
  | { k: "role"; name: string } // a named state/hyperparameter input
  | { k: "c"; v: number } // a constant
  | { k: "u"; op: OptUnary; a: OptTerm }
  | { k: "b"; op: OptBinary; a: OptTerm; b: OptTerm }
  // The CONTRACTION node (Crystal 3 tail): a 2D matmul `A·B` with per-operand
  // transpose flags — the smallest contraction form Newton–Schulz needs
  // (`X·Xᵀ`, `A·A`, `B·X`). The interpreter emits `rt.matmul` (the existing
  // kernel, REFERENCED not re-owned — design §4.4, category d). This is the ONE
  // node the pure-elementwise optimizer frame lacked (the P3 contraction the
  // §16 MUON deferral named); every leaf stays DATA (booleans + child terms).
  | { k: "mm"; a: OptTerm; b: OptTerm; ta: boolean; tb: boolean };

// Builders (data, never closures).
export const role = (name: string): OptTerm => ({ k: "role", name });
export const konst = (v: number): OptTerm => ({ k: "c", v });
const u = (op: OptUnary, a: OptTerm): OptTerm => ({ k: "u", op, a });
const bin = (op: OptBinary, a: OptTerm, b: OptTerm): OptTerm => ({
  k: "b",
  op,
  a,
  b,
});
export const oNeg = (a: OptTerm) => u("neg", a);
export const oSqrt = (a: OptTerm) => u("sqrt", a);
export const oSign = (a: OptTerm) => u("sign", a);
export const oAbs = (a: OptTerm) => u("abs", a);
export const oExp = (a: OptTerm) => u("exp", a);
export const oAdd = (a: OptTerm, b: OptTerm) => bin("add", a, b);
export const oSub = (a: OptTerm, b: OptTerm) => bin("sub", a, b);
export const oMul = (a: OptTerm, b: OptTerm) => bin("mul", a, b);
export const oDiv = (a: OptTerm, b: OptTerm) => bin("div", a, b);
/**
 * The 2D contraction builder `A·B` with optional per-operand transpose. The
 * ONLY non-elementwise primitive; `oMatmul(X, X, { tb: true })` is `X·Xᵀ`, the
 * Newton–Schulz kernel of Muon. Data in, data out — the interpreter realizes it
 * as `rt.matmul` over the runtime.
 */
export const oMatmul = (
  a: OptTerm,
  b: OptTerm,
  opts?: { ta?: boolean; tb?: boolean },
): OptTerm => ({ k: "mm", a, b, ta: opts?.ta ?? false, tb: opts?.tb ?? false });

// ----------------------------------------------------------------------------
// The program schema — a set of in-place state updates + the param update, each
// a pure OptTerm. The realizer sequences the copy_ effects (design §4.6).
// ----------------------------------------------------------------------------

export interface StateUpdate {
  /** The persistent per-param state slot this term writes (e.g. "m", "v"). */
  slot: string;
  /** The update term — over the PRE-step roles (old state + g + hypers). */
  expr: OptTerm;
}

export interface OptimizerProgram {
  name: string;
  /** Persistent per-param state slots (besides the param), e.g. ["m","v"]. */
  state: readonly string[];
  /** In-place state updates. Each `expr` reads the OLD state (design §4.6). */
  stateUpdates: readonly StateUpdate[];
  /**
   * The new parameter value `p'` as a term. Which state it reads (pre- vs
   * post-update) is a REALIZER decision expressed by role binding: Adam/SGD
   * bind `m`/`v` to the POST-copy_ state (p' reads the updated moments, matching
   * PyTorch + preserving the dangling-copy_ lifetime discipline); Lion binds `m`
   * to the OLD state (its step direction is a β1-interpolation of the old
   * momentum, distinct from the stored β2-EMA).
   */
  paramUpdate: OptTerm;
  /** Hyperparameter roles the program reads (documentation + the schema gate). */
  hyperRoles: readonly string[];
}

// ----------------------------------------------------------------------------
// The structural schema gate — a program is DATA (the `assertNoDefinitionBody`
// analogue for the optimizer frame; covenant/R22 defense).
// ----------------------------------------------------------------------------

const OPT_TERM_KINDS = new Set(["role", "c", "u", "b", "mm"]);
const OPT_UNARY = new Set<string>(["neg", "sqrt", "sign", "abs", "exp"]);
const OPT_BINARY = new Set<string>(["add", "sub", "mul", "div"]);

function assertNoOptTermBody(t: OptTerm, path: string): void {
  const walk = (v: unknown, p: string): void => {
    if (v === null || v === undefined)
      throw new Error(`OptimizerProgram schema gate: ${p} is null/undefined.`);
    const ty = typeof v;
    if (ty === "function")
      throw new Error(
        `OptimizerProgram schema gate: ${p} is a function — the program must be DATA (no embedded op body).`,
      );
    if (ty === "string" || ty === "number" || ty === "boolean") return;
    if (ty === "object") {
      if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer)
        throw new Error(
          `OptimizerProgram schema gate: ${p} is a buffer leaf — a term names roles/primitives, never buffers.`,
        );
      const kind = (v as { k?: unknown }).k;
      if (typeof kind !== "string" || !OPT_TERM_KINDS.has(kind))
        throw new Error(
          `OptimizerProgram schema gate: ${p} has kind ${JSON.stringify(kind)} which is not a term primitive.`,
        );
      if (kind === "u" && !OPT_UNARY.has((v as { op: string }).op))
        throw new Error(
          `OptimizerProgram schema gate: ${p} has unknown unary op ${JSON.stringify((v as { op: string }).op)}.`,
        );
      if (kind === "b" && !OPT_BINARY.has((v as { op: string }).op))
        throw new Error(
          `OptimizerProgram schema gate: ${p} has unknown binary op ${JSON.stringify((v as { op: string }).op)}.`,
        );
      for (const [key, val] of Object.entries(v as Record<string, unknown>)) {
        if (key === "k" || key === "op" || key === "name") continue;
        walk(val, `${p}.${key}`);
      }
      return;
    }
    throw new Error(
      `OptimizerProgram schema gate: ${p} has non-data type ${ty}.`,
    );
  };
  walk(t, path);
}

/** Prove an optimizer program is DATA (schema gate). */
export function assertNoOptimizerProgramBody(prog: OptimizerProgram): void {
  for (const su of prog.stateUpdates) {
    if (!prog.state.includes(su.slot))
      throw new Error(
        `OptimizerProgram schema gate: ${prog.name} updates unknown state slot '${su.slot}'.`,
      );
    assertNoOptTermBody(su.expr, `${prog.name}.state[${su.slot}]`);
  }
  assertNoOptTermBody(prog.paramUpdate, `${prog.name}.param`);
}

// ----------------------------------------------------------------------------
// The interpreter — realize an OptTerm over the RuntimeEngine. Roles bind to the
// live state/hyper tensors (or scalars); primitives dispatch to `rt.*`. This is
// the SINGLE SOURCE for the moment/param arithmetic: the foreach + elementwise
// paths call it, so the hand chains delete. `number×number` folds in JS so a
// pure-scalar sub-term (e.g. `1-β1`) stays a constant, not a 0-d graph tensor —
// keeping the generated graph byte-identical to the hand code it replaces.
// ----------------------------------------------------------------------------

export type OptVal = number | RuntimeTensor;

export interface OptRoles {
  [name: string]: OptVal;
}

function needRole(roles: OptRoles, name: string): OptVal {
  const v = roles[name];
  if (v === undefined)
    throw new Error(
      `evalOptTerm: program references role '${name}' but it was not provided.`,
    );
  return v;
}

/**
 * Interpret an OptTerm over the runtime engine, producing a graph value.
 *
 * `sink`, if given, collects every intermediate tensor this eval CREATES (never
 * the role inputs) so the realizer can dispose the packed full-model-size
 * intermediates it generated — preserving the foreach path's exact memory
 * discipline (the arithmetic derives; the disposal effect stays the realizer's).
 */
export function evalOptTerm(
  t: OptTerm,
  rt: RuntimeEngine,
  roles: OptRoles,
  sink?: RuntimeTensor[],
  memo?: Map<OptTerm, OptVal>,
): OptVal {
  // Memoize by term-object IDENTITY: a subterm shared by object reuse is
  // realized ONCE. Muon's Newton-Schulz reuses the same X/A objects ~8× per
  // step, so a naive tree walk explodes to ~8^S matmul dispatches (a 32GB GPU
  // blowup at S=5); memoization collapses the interpretation to the term's DAG
  // size. Sharing an OptTerm object is the authoring signal that its value is
  // shared; distinct objects (structurally equal or not) still evaluate apart.
  const cache = memo ?? new Map<OptTerm, OptVal>();
  const cached = cache.get(t);
  if (cached !== undefined) return cached;
  const result = evalOptTermCore(t, rt, roles, sink, cache);
  cache.set(t, result);
  return result;
}

function evalOptTermCore(
  t: OptTerm,
  rt: RuntimeEngine,
  roles: OptRoles,
  sink: RuntimeTensor[] | undefined,
  cache: Map<OptTerm, OptVal>,
): OptVal {
  const track = (v: OptVal): OptVal => {
    if (sink !== undefined && typeof v !== "number") sink.push(v);
    return v;
  };
  switch (t.k) {
    case "role":
      return needRole(roles, t.name);
    case "c":
      return t.v;
    case "u": {
      const a = evalOptTerm(t.a, rt, roles, sink, cache);
      if (typeof a === "number") {
        // Fold pure-scalar unaries in JS.
        switch (t.op) {
          case "neg":
            return -a;
          case "sqrt":
            return Math.sqrt(a);
          case "sign":
            return Math.sign(a);
          case "abs":
            return Math.abs(a);
          case "exp":
            return Math.exp(a);
        }
      }
      switch (t.op) {
        case "neg":
          return track(rt.neg(a));
        case "sqrt":
          return track(rt.sqrt(a));
        case "sign":
          return track(rt.sign(a));
        case "abs":
          return track(rt.abs(a));
        case "exp":
          return track(rt.exp(a));
      }
      break;
    }
    case "b": {
      const a = evalOptTerm(t.a, rt, roles, sink, cache);
      const b = evalOptTerm(t.b, rt, roles, sink, cache);
      if (typeof a === "number" && typeof b === "number") {
        // Fold pure-scalar binaries in JS (keeps `1-β1` a constant).
        switch (t.op) {
          case "add":
            return a + b;
          case "sub":
            return a - b;
          case "mul":
            return a * b;
          case "div":
            return a / b;
        }
      }
      switch (t.op) {
        case "add":
          return track(rt.add(a, b));
        case "sub":
          return track(rt.sub(a, b));
        case "mul":
          return track(rt.mul(a, b));
        case "div":
          return track(rt.div(a, b));
      }
      break;
    }
    case "mm": {
      // The contraction node — both operands MUST be 2D tensors (no scalar
      // fold; a matmul over scalars is meaningless). Transposes are 2D views
      // (dim0↔dim1), REFERENCED into the existing rt.matmul kernel (§4.4).
      const a = evalOptTerm(t.a, rt, roles, sink, cache);
      const b = evalOptTerm(t.b, rt, roles, sink, cache);
      if (typeof a === "number" || typeof b === "number")
        throw new Error(
          "evalOptTerm: a contraction (mm) node requires tensor operands, but a leaf folded to a scalar.",
        );
      const aM = t.ta ? track(rt.transpose(a, { dim0: 0, dim1: 1 })) : a;
      const bM = t.tb ? track(rt.transpose(b, { dim0: 0, dim1: 1 })) : b;
      return track(rt.matmul(aM, bM));
    }
  }
  throw new Error(`evalOptTerm: unhandled term ${JSON.stringify(t)}`);
}

/** Interpret a term that MUST reduce to a tensor (a state/param update). */
export function evalOptTensor(
  t: OptTerm,
  rt: RuntimeEngine,
  roles: OptRoles,
  sink?: RuntimeTensor[],
): RuntimeTensor {
  const v = evalOptTerm(t, rt, roles, sink);
  if (typeof v === "number")
    throw new Error(
      "evalOptTensor: an optimizer update must be a tensor, but the term folded to a scalar.",
    );
  return v;
}

// ============================================================================
// The catalog — AdamW / SGD+momentum / Lion as sibling programs (design §4.6).
// ============================================================================

// Shared role handles.
const M = role("m");
const V = role("v");
const G = role("g");
const P = role("p");
const LR = role("lr");
const EPS = role("eps");
const WD = role("wd");
const BETA1 = role("beta1");
const OM_BETA1 = role("om_beta1"); // 1 − β1
const BETA2 = role("beta2");
const OM_BETA2 = role("om_beta2"); // 1 − β2
const BC1 = role("bc1"); // bias correction 1 = 1 − β1^t
const BC2 = role("bc2"); // bias correction 2 = 1 − β2^t

/**
 * AdamW (the design's canonical example, §1 P5).
 *
 *   m' = β1·m + (1−β1)·g
 *   v' = β2·v + (1−β2)·g²
 *   m̂ = m'/bc1 ; v̂ = v'/bc2                    (bias-correction as DATA — the
 *                                                 step rides in as bc1/bc2, the
 *                                                 volatile-uniform discipline)
 *   p' = p − ( lr·m̂/(√v̂ + ε) + lr·wd·p )        (decoupled weight decay)
 *
 * The `p'` term reads `m`/`v` bound to the POST-copy_ state (the realizer's
 * dangling-copy_ lifetime discipline). Classic-Adam (L2) is the SAME program:
 * the realizer folds wd into `g` (g += wd·p) and binds `wd`=0 here, so the
 * decoupled term vanishes to a byte-identical `+0` — one program, two policies
 * selected by role binding (matching the fused kernel's `decoupledWd` branch).
 *
 * BYTE-IDENTICAL BY CONSTRUCTION: this term produces the exact graph the hand
 * foreach/elementwise chains produced (same op order, same constant placement),
 * so the fused-vs-elementwise / oracle / 124M gates hold unchanged.
 */
export const ADAMW_M_NEW = oAdd(oMul(M, BETA1), oMul(G, OM_BETA1));
export const ADAMW_V_NEW = oAdd(oMul(V, BETA2), oMul(oMul(G, G), OM_BETA2));
const ADAMW_MHAT = oDiv(M, BC1);
const ADAMW_VHAT = oDiv(V, BC2);
const ADAMW_DENOM = oAdd(oSqrt(ADAMW_VHAT), EPS);
/**
 * The update magnitude `lr·m̂/(√v̂+ε)` — the heavy arithmetic core, exported so
 * the realizer can derive it while keeping the decoupled-wd term CONDITIONAL:
 * an unconditional `+ lr·wd·p` term would materialize a full-model-size `+0`
 * intermediate in the (common) wd=0 path (a 124M memory regression). The
 * realizer folds wd in only when it fires; the full `p'` (below) is the complete
 * composition the reference gate + fused-kernel assertion check.
 */
export const ADAMW_SCALED = oMul(oDiv(ADAMW_MHAT, ADAMW_DENOM), LR);
const ADAMW_WD_TERM = oMul(P, oMul(LR, WD)); // decoupled; 0 when wd=0
export const ADAMW_P_NEW = oSub(P, oAdd(ADAMW_SCALED, ADAMW_WD_TERM));

export const ADAMW_PROGRAM: OptimizerProgram = {
  name: "adamw",
  state: ["m", "v"],
  stateUpdates: [
    { slot: "m", expr: ADAMW_M_NEW },
    { slot: "v", expr: ADAMW_V_NEW },
  ],
  paramUpdate: ADAMW_P_NEW,
  hyperRoles: [
    "lr",
    "eps",
    "wd",
    "beta1",
    "om_beta1",
    "beta2",
    "om_beta2",
    "bc1",
    "bc2",
  ],
};

/**
 * SGD with momentum (a SIBLING definition — the generality proof).
 *
 *   v' = μ·v + g          (g already carries L2 weight decay: g += wd·p)
 *   p' = p − lr·v'
 *
 * DEFINED + reference-asserted here (test/semantic-optimizer.spec.ts). SGD's
 * realizer (src/optim/sgd.ts) keeps its lr-rides-`sub`-alpha delivery (a
 * realizer policy the LR-schedule scalar-table path depends on) — the same
 * derive-vs-assert split the fused Adam kernel gets: the program is the SOURCE,
 * the realizer is a realization the differential pins to it.
 */
const MU = role("mu"); // momentum μ
const SGD_V_NEW = oAdd(oMul(V, MU), G);
const SGD_P_NEW = oSub(P, oMul(V, LR));

export const SGD_MOMENTUM_PROGRAM: OptimizerProgram = {
  name: "sgd_momentum",
  state: ["v"],
  stateUpdates: [{ slot: "v", expr: SGD_V_NEW }],
  paramUpdate: SGD_P_NEW,
  hyperRoles: ["lr", "mu"],
};

/** Plain SGD (no momentum): p' = p − lr·g. */
export const SGD_PROGRAM: OptimizerProgram = {
  name: "sgd",
  state: [],
  stateUpdates: [],
  paramUpdate: oSub(P, oMul(G, LR)),
  hyperRoles: ["lr"],
};

/**
 * LION (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms") — the
 * generality DIVIDEND: a sign-based optimizer that derives from the SAME algebra
 * with no hand kernel and no hand grads.
 *
 *   c   = β1·m + (1−β1)·g        (the step direction — a β1 interpolation)
 *   p'  = p − lr·( sign(c) + wd·p )   (decoupled weight decay)
 *   m'  = β2·m + (1−β2)·g        (the stored EMA — a DIFFERENT, β2, average)
 *
 * The step direction `c` reads the OLD momentum (the realizer binds `m` to the
 * pre-update state for `paramUpdate`); the stored `m'` is a separate β2-EMA.
 * This is exactly the design's claim that Lion is "~free from the algebra".
 */
const LION_C = oAdd(oMul(M, BETA1), oMul(G, OM_BETA1)); // reads OLD m
export const LION_M_NEW = oAdd(oMul(M, BETA2), oMul(G, OM_BETA2)); // reads OLD m
/** The signed step magnitude `lr·sign(c)` — exported so the realizer derives it
 *  while keeping the decoupled-wd term CONDITIONAL (no full-size `+0` at wd=0). */
export const LION_STEP = oMul(oSign(LION_C), LR);
const LION_WD_TERM = oMul(P, oMul(LR, WD));
export const LION_P_NEW = oSub(P, oAdd(LION_STEP, LION_WD_TERM));

export const LION_PROGRAM: OptimizerProgram = {
  name: "lion",
  state: ["m"],
  stateUpdates: [{ slot: "m", expr: LION_M_NEW }],
  paramUpdate: LION_P_NEW,
  hyperRoles: ["lr", "wd", "beta1", "om_beta1", "beta2", "om_beta2"],
};

/**
 * MUON — the CONTRACTION dividend (Crystal 3 tail; design §2 category d, §16
 * deferral CASHED). Muon's core is Newton–Schulz orthogonalization of the
 * momentum matrix — a fixed-point of MATMUL CONTRACTIONS — which the pure-
 * elementwise frame lacked. With the `mm` node it becomes a program, exactly as
 * Lion did from the elementwise algebra: an optimizer FROM ITS DEFINITION.
 *
 *   buf' = μ·buf + g                                    (momentum, SGD-form state)
 *   X₀   = buf · ns_scale        (ns_scale = 1/(‖buf‖_F+ε), realizer-computed —
 *                                 normalizes into the NS convergence basin)
 *   for k in 1..S:   A = X·Xᵀ ;  X ← a·X + (b·A + c·A·A)·X   (quintic, coeffs DATA)
 *   p'   = p − lr·( rms·X_S + wd·p )    (rms = √max(1,rows/cols); decoupled wd)
 *
 * The Newton–Schulz coefficients (Jordan et al., the published quintic) and the
 * iteration COUNT are DATA; the iteration is a fully UNROLLED composition. The
 * orientation transpose (run NS with rows≤cols) and the AdamW routing of
 * embedding/1D params are the REALIZER's declared POLICY (src/optim/muon.ts) —
 * the arithmetic derives, the routing/effect is declared (§4.6).
 */
// The published Newton–Schulz quintic coefficients (Jordan et al.) — DATA.
export const MUON_NS_COEFFS = { a: 3.4445, b: -4.775, c: 2.0315 } as const;
export const MUON_NS_STEPS = 5;

const NS_SCALE = role("ns_scale"); // 1/(‖buf‖_F + ε) — realizer-computed (bc1 precedent)
const RMS = role("rms"); // √max(1, rows/cols) — the shape-magnitude match, DATA

/** One quintic Newton–Schulz iteration: X ← a·X + (b·A + c·A·A)·X with A = X·Xᵀ. */
function nsStep(X: OptTerm): OptTerm {
  const A = oMatmul(X, X, { tb: true }); // X·Xᵀ  (the Gram contraction)
  const A2 = oMatmul(A, A); // A·A
  const B = oAdd(
    oMul(A, konst(MUON_NS_COEFFS.b)),
    oMul(A2, konst(MUON_NS_COEFFS.c)),
  );
  return oAdd(oMul(X, konst(MUON_NS_COEFFS.a)), oMatmul(B, X)); // a·X + B·X
}

/**
 * The orthogonalization term: `steps` quintic iterations over the normalized
 * momentum `m·ns_scale`. The COUNT is DATA — the composition is fully UNROLLED
 * (S contractions per step), so a program is a finite term, never a loop body.
 */
export function buildMuonOrtho(steps: number = MUON_NS_STEPS): OptTerm {
  let X: OptTerm = oMul(M, NS_SCALE); // normalize into the NS convergence basin
  for (let i = 0; i < steps; i++) X = nsStep(X);
  return X;
}

/** buf' = μ·buf + g — the momentum buffer (the SGD-momentum state form). */
export const MUON_M_NEW = oAdd(oMul(M, MU), G);
/** The default (5-step) orthogonalization of the momentum buffer. */
export const MUON_ORTHO = buildMuonOrtho();
/** The magnitude-matched update: rms·orthogonalize(buf). */
export const MUON_UPDATE = oMul(MUON_ORTHO, RMS);
const MUON_WD_TERM = oMul(P, oMul(LR, WD));
/** p' = p − lr·(rms·orthogonalize(buf) + wd·p) — the complete term (the gate source). */
export const MUON_P_NEW = oSub(P, oAdd(oMul(MUON_UPDATE, LR), MUON_WD_TERM));

/**
 * The realizer's apply tail: p' from a PRECOMPUTED (back-oriented, rms-scaled)
 * update `u`. The orientation transpose is a shape POLICY that sits BETWEEN the
 * orthogonalization and this apply, so muon.ts interprets the two pieces around
 * the transpose rather than MUON_P_NEW whole — the ADAMW_SCALED-vs-ADAMW_P_NEW
 * precedent (both are DATA; the full term stays the single reference source).
 */
const MUON_U = role("u");
export const MUON_P_APPLY = oSub(
  P,
  oAdd(oMul(MUON_U, LR), MUON_WD_TERM),
);

export const MUON_PROGRAM: OptimizerProgram = {
  name: "muon",
  state: ["m"],
  stateUpdates: [{ slot: "m", expr: MUON_M_NEW }],
  paramUpdate: MUON_P_NEW,
  hyperRoles: ["lr", "wd", "mu", "ns_scale", "rms"],
};

/** The optimizer-program catalog (the DEFINED programs; MUON now among them). */
export const OPTIMIZER_PROGRAMS: readonly OptimizerProgram[] = [
  ADAMW_PROGRAM,
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
  LION_PROGRAM,
  MUON_PROGRAM,
];
