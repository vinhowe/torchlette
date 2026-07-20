/**
 * CRYSTAL CAMPAIGN 3 — Derive-the-Reference feasibility probes.
 *
 * A self-contained probe (no GPU) for the design doc's spine:
 *   Probe 2 (adjoint):   define the elementwise family as expression trees,
 *                        derive gradients mechanically via adjoint rules, and
 *                        compare the derived backward numerically against the
 *                        hand-written gradient tables (src/ops/registry.ts).
 *   Probe 3 (reference): interpret the SAME definitions as a CPU reference and
 *                        compare byte-for-byte against the hand-written CPU op
 *                        bodies (src/backend/cpu/numeric.ts).
 *   Probe 4 (composite): define RMSNorm as a composition of primitives and
 *                        check the composition's reference eval against an
 *                        independent direct implementation of the semantics.
 *
 * The CPU op bodies compute in f64 and round once on store into a Float32Array
 * (numeric.ts discipline), so this probe does the same: evaluate in f64, round
 * to f32 at the comparison boundary via Float32Array. "byte-identical" = equal
 * f32 bit patterns; we also report the max abs/ulp gap for near-misses so an
 * algebraic-rewrite difference is not mistaken for a bug.
 */

// ----------------------------------------------------------------------------
// Expression tree — the single semantic definition schema (formula as data)
// ----------------------------------------------------------------------------

type Expr =
  | { k: "x" } // primary input (unary op input, or binary left operand a)
  | { k: "y" } // second input (binary right operand b)
  | { k: "g" } // upstream gradient (used only in gradient expressions)
  | { k: "c"; v: number } // constant
  | { k: "neg" | "exp" | "log" | "sqrt" | "sin" | "cos" | "tanh" | "abs" | "sign" | "recip"; a: Expr }
  | { k: "add" | "sub" | "mul" | "div" | "pow" | "min" | "max"; a: Expr; b: Expr }
  | { k: "gt" | "ge" | "lt" | "le"; a: Expr; b: Expr } // 1.0 / 0.0
  | { k: "where"; c: Expr; a: Expr; b: Expr };

const x: Expr = { k: "x" };
const y: Expr = { k: "y" };
const g: Expr = { k: "g" };
const c = (v: number): Expr => ({ k: "c", v });
const neg = (a: Expr): Expr => ({ k: "neg", a });
const exp = (a: Expr): Expr => ({ k: "exp", a });
const log = (a: Expr): Expr => ({ k: "log", a });
const sqrt = (a: Expr): Expr => ({ k: "sqrt", a });
const sin = (a: Expr): Expr => ({ k: "sin", a });
const cos = (a: Expr): Expr => ({ k: "cos", a });
const tanh = (a: Expr): Expr => ({ k: "tanh", a });
const abs = (a: Expr): Expr => ({ k: "abs", a });
const sign = (a: Expr): Expr => ({ k: "sign", a });
const recip = (a: Expr): Expr => ({ k: "recip", a });
const add = (a: Expr, b: Expr): Expr => ({ k: "add", a, b });
const sub = (a: Expr, b: Expr): Expr => ({ k: "sub", a, b });
const mul = (a: Expr, b: Expr): Expr => ({ k: "mul", a, b });
const div = (a: Expr, b: Expr): Expr => ({ k: "div", a, b });
const powE = (a: Expr, b: Expr): Expr => ({ k: "pow", a, b });
const minE = (a: Expr, b: Expr): Expr => ({ k: "min", a, b });
const maxE = (a: Expr, b: Expr): Expr => ({ k: "max", a, b });
const gt = (a: Expr, b: Expr): Expr => ({ k: "gt", a, b });
const ge = (a: Expr, b: Expr): Expr => ({ k: "ge", a, b });
const lt = (a: Expr, b: Expr): Expr => ({ k: "lt", a, b });
const le = (a: Expr, b: Expr): Expr => ({ k: "le", a, b });
const where = (cnd: Expr, a: Expr, b: Expr): Expr => ({ k: "where", c: cnd, a, b });

// ----------------------------------------------------------------------------
// Evaluator (f64 internally, matching numeric.ts store-once-round discipline)
// ----------------------------------------------------------------------------

interface Env {
  x: number;
  y: number;
  g: number;
}

function evalExpr(e: Expr, env: Env): number {
  switch (e.k) {
    case "x":
      return env.x;
    case "y":
      return env.y;
    case "g":
      return env.g;
    case "c":
      return e.v;
    case "neg":
      return -evalExpr(e.a, env);
    case "exp":
      return Math.exp(evalExpr(e.a, env));
    case "log":
      return Math.log(evalExpr(e.a, env));
    case "sqrt":
      return Math.sqrt(evalExpr(e.a, env));
    case "sin":
      return Math.sin(evalExpr(e.a, env));
    case "cos":
      return Math.cos(evalExpr(e.a, env));
    case "tanh":
      return Math.tanh(evalExpr(e.a, env));
    case "abs":
      return Math.abs(evalExpr(e.a, env));
    case "sign":
      return Math.sign(evalExpr(e.a, env));
    case "recip":
      return 1.0 / evalExpr(e.a, env);
    case "add":
      return evalExpr(e.a, env) + evalExpr(e.b, env);
    case "sub":
      return evalExpr(e.a, env) - evalExpr(e.b, env);
    case "mul":
      return evalExpr(e.a, env) * evalExpr(e.b, env);
    case "div":
      return evalExpr(e.a, env) / evalExpr(e.b, env);
    case "pow":
      return Math.pow(evalExpr(e.a, env), evalExpr(e.b, env));
    case "min":
      return Math.min(evalExpr(e.a, env), evalExpr(e.b, env));
    case "max":
      return Math.max(evalExpr(e.a, env), evalExpr(e.b, env));
    case "gt":
      return evalExpr(e.a, env) > evalExpr(e.b, env) ? 1.0 : 0.0;
    case "ge":
      return evalExpr(e.a, env) >= evalExpr(e.b, env) ? 1.0 : 0.0;
    case "lt":
      return evalExpr(e.a, env) < evalExpr(e.b, env) ? 1.0 : 0.0;
    case "le":
      return evalExpr(e.a, env) <= evalExpr(e.b, env) ? 1.0 : 0.0;
    case "where":
      return evalExpr(e.c, env) !== 0 ? evalExpr(e.a, env) : evalExpr(e.b, env);
  }
}

// ----------------------------------------------------------------------------
// Adjoint rules as data — d(expr)/d(v) for v in {"x","y"}
// ----------------------------------------------------------------------------

function deriv(e: Expr, wrt: "x" | "y"): Expr {
  switch (e.k) {
    case "x":
      return wrt === "x" ? c(1) : c(0);
    case "y":
      return wrt === "y" ? c(1) : c(0);
    case "g":
    case "c":
      return c(0);
    case "neg":
      return neg(deriv(e.a, wrt));
    case "exp":
      return mul(exp(e.a), deriv(e.a, wrt)); // d e^u = e^u u'
    case "log":
      return div(deriv(e.a, wrt), e.a); // d ln u = u'/u
    case "sqrt":
      return div(deriv(e.a, wrt), mul(c(2), sqrt(e.a))); // u'/(2√u)
    case "sin":
      return mul(cos(e.a), deriv(e.a, wrt));
    case "cos":
      return mul(neg(sin(e.a)), deriv(e.a, wrt));
    case "tanh":
      return mul(sub(c(1), mul(tanh(e.a), tanh(e.a))), deriv(e.a, wrt));
    case "abs":
      return mul(sign(e.a), deriv(e.a, wrt));
    case "sign":
      return c(0);
    case "recip":
      return neg(div(deriv(e.a, wrt), mul(e.a, e.a))); // -u'/u²
    case "add":
      return add(deriv(e.a, wrt), deriv(e.b, wrt));
    case "sub":
      return sub(deriv(e.a, wrt), deriv(e.b, wrt));
    case "mul":
      return add(mul(deriv(e.a, wrt), e.b), mul(e.a, deriv(e.b, wrt)));
    case "div":
      return div(
        sub(mul(deriv(e.a, wrt), e.b), mul(e.a, deriv(e.b, wrt))),
        mul(e.b, e.b),
      );
    case "pow": {
      // General pow is only differentiated here for a constant exponent b (the
      // registry's tsGrad path). d/da a^n = n·a^(n-1). Variable-exponent pow is
      // out of the elementwise adjoint scope (needs a^b·ln a — covered in notes).
      if (e.b.k === "c" && wrt === "x") {
        return mul(mul(c(e.b.v), powE(e.a, c(e.b.v - 1))), deriv(e.a, wrt));
      }
      return c(NaN); // signal: not handled by the pure-elementwise adjoint
    }
    case "min":
      // d min(a,b)/dv = (a<=b) ? da : db
      return where(le(e.a, e.b), deriv(e.a, wrt), deriv(e.b, wrt));
    case "max":
      return where(ge(e.a, e.b), deriv(e.a, wrt), deriv(e.b, wrt));
    case "gt":
    case "ge":
    case "lt":
    case "le":
      return c(0); // comparisons: subgradient 0
    case "where":
      return where(e.c, deriv(e.a, wrt), deriv(e.b, wrt));
  }
}

/** The derived vector-Jacobian product for a unary op: g · dOut/dx. */
const vjpUnary = (def: Expr): Expr => mul(g, deriv(def, "x"));
/** The derived VJPs for a binary op wrt each operand. */
const vjpBinaryA = (def: Expr): Expr => mul(g, deriv(def, "x"));
const vjpBinaryB = (def: Expr): Expr => mul(g, deriv(def, "y"));

// ----------------------------------------------------------------------------
// The definitions (single source) and the hand-written surfaces to check against
// ----------------------------------------------------------------------------

// Unary definitions — EXACTLY the formulas in src/backend/cpu/numeric.ts UNARY_OPS.
const UNARY_DEFS: Record<string, Expr> = {
  sqrt: sqrt(x),
  exp: exp(x),
  log: log(x),
  neg: neg(x),
  abs: abs(x),
  tanh: tanh(x),
  sigmoid: recip(add(c(1), exp(neg(x)))), // 1/(1+e^-x)
  silu: div(x, add(c(1), exp(neg(x)))), // x/(1+e^-x)
  sin: sin(x),
  cos: cos(x),
  rsqrt: recip(sqrt(x)), // 1/√x
  sign: sign(x),
  relu: where(gt(x, c(0)), x, c(0)), // x>0 ? x : 0
  // floor/ceil/round/isfinite: not expressible as smooth primitives — see notes.
};

// Hand-written CPU unary bodies (transcribed from numeric.ts UNARY_OPS) for the
// reference probe. These are the ground truth we must reproduce.
const HAND_UNARY: Record<string, (x: number) => number> = {
  sqrt: Math.sqrt,
  exp: Math.exp,
  log: Math.log,
  neg: (x) => -x,
  abs: Math.abs,
  tanh: Math.tanh,
  sigmoid: (x) => 1.0 / (1.0 + Math.exp(-x)),
  silu: (x) => x / (1.0 + Math.exp(-x)),
  sin: Math.sin,
  cos: Math.cos,
  rsqrt: (x) => 1.0 / Math.sqrt(x),
  sign: Math.sign,
  relu: (x) => (x > 0 ? x : 0),
};

// Hand-written table gradients (transcribed from src/ops/registry.ts) — the
// surface the ADJOINT-derived VJP must reproduce. Written as Expr so we can
// evaluate them identically. null = non-differentiable in the table.
const HAND_UNARY_GRAD: Record<string, Expr | null> = {
  // relu: g * (x > 0)
  relu: mul(g, gt(x, c(0))),
  // silu: g * (sig + x·sig·(1-sig)), sig = sigmoid(x)
  silu: mul(
    g,
    add(
      recip(add(c(1), exp(neg(x)))),
      mul(
        x,
        mul(
          recip(add(c(1), exp(neg(x)))),
          sub(c(1), recip(add(c(1), exp(neg(x))))),
        ),
      ),
    ),
  ),
  // sigmoid: sig·(1-sig)·g
  sigmoid: mul(
    mul(
      recip(add(c(1), exp(neg(x)))),
      sub(c(1), recip(add(c(1), exp(neg(x))))),
    ),
    g,
  ),
  // tanh: (1 - t²)·g, t = tanh(x)
  tanh: mul(sub(c(1), mul(tanh(x), tanh(x))), g),
  // neg: -g
  neg: neg(g),
  // abs: g·sign(x)
  abs: mul(g, sign(x)),
  // exp: g·exp(x)
  exp: mul(g, exp(x)),
  // log: g / (x + 1e-8)   <-- note the epsilon in the table
  log: div(g, add(x, c(1e-8))),
  // sqrt: g · 0.5/(√x + 1e-8)   <-- note the epsilon
  sqrt: mul(g, div(c(0.5), add(sqrt(x), c(1e-8)))),
  // rsqrt: g · (-0.5·r·r·r), r = rsqrt(x)
  rsqrt: mul(g, mul(c(-0.5), mul(recip(sqrt(x)), mul(recip(sqrt(x)), recip(sqrt(x)))))),
  // sin: g·cos(x)
  sin: mul(g, cos(x)),
  // cos: g·(-sin(x))
  cos: mul(g, neg(sin(x))),
  // sign/floor/ceil/round/isfinite: grad = null
  sign: null,
};

// Binary definitions (numeric.ts BINARY_OPS + sub/div).
const BINARY_DEFS: Record<string, Expr> = {
  add: add(x, y),
  sub: sub(x, y),
  mul: mul(x, y),
  div: div(x, y),
  minimum: minE(x, y),
  maximum: maxE(x, y),
};

const HAND_BINARY: Record<string, (x: number, y: number) => number> = {
  add: (x, y) => x + y,
  sub: (x, y) => x - y,
  mul: (x, y) => x * y,
  div: (x, y) => x / y,
  minimum: Math.min,
  maximum: Math.max,
};

// Table binary gradients (ttGrad), transcribed from registry.ts, as [dA, dB] Exprs.
const HAND_BINARY_GRAD: Record<string, [Expr, Expr] | null> = {
  add: [g, g],
  sub: null, // registry has no ttGrad for sub (handled structurally) — see notes
  mul: [mul(g, y), mul(g, x)],
  div: [div(g, y), mul(g, div(neg(x), mul(y, y)))],
  minimum: [mul(g, le(x, y)), mul(g, gt(x, y))],
  maximum: [mul(g, ge(x, y)), mul(g, lt(x, y))],
};

// ----------------------------------------------------------------------------
// Comparison helpers — f32 byte identity + gap reporting
// ----------------------------------------------------------------------------

const f32 = (v: number): number => Math.fround(v);
const F = new Float32Array(1);
const U = new Uint32Array(F.buffer);
const bits = (v: number): number => {
  F[0] = v;
  return U[0];
};

function ulpGap(a: number, b: number): number {
  const fa = f32(a);
  const fb = f32(b);
  if (fa === fb) return 0;
  if (Number.isNaN(fa) && Number.isNaN(fb)) return 0;
  if (!Number.isFinite(fa) || !Number.isFinite(fb)) return Infinity;
  return Math.abs((bits(fa) | 0) - (bits(fb) | 0));
}

interface CompareResult {
  op: string;
  n: number;
  byteExact: number; // count of sample points with identical f32 bits
  maxAbs: number;
  maxUlp: number;
  worst?: { in: number[]; a: number; b: number };
}

const UNARY_SAMPLES = [
  -8, -3.5, -1.0, -0.5, -0.0001, 0.0, 0.0001, 0.3, 0.7, 1.0, 2.5, 5.0, 7.3, 12.0,
];
const BINARY_SAMPLES: [number, number][] = [];
for (const a of [-5, -1.3, -0.2, 0.0, 0.4, 1.1, 3.7]) {
  for (const b of [-4.2, -0.7, 0.5, 1.9, 6.1]) BINARY_SAMPLES.push([a, b]);
}
const GRAD_UPSTREAM = [1.0, -2.3, 0.5];

function compareUnary(
  op: string,
  produce: (xv: number) => number,
  ground: (xv: number) => number,
  domainOk: (xv: number) => boolean = () => true,
): CompareResult {
  let byteExact = 0,
    n = 0,
    maxAbs = 0,
    maxUlp = 0;
  let worst: CompareResult["worst"];
  for (const xv of UNARY_SAMPLES) {
    if (!domainOk(xv)) continue;
    n++;
    const a = f32(produce(xv));
    const b = f32(ground(xv));
    if (bits(a) === bits(b) || (Number.isNaN(a) && Number.isNaN(b))) byteExact++;
    const absd = Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) : a === b ? 0 : Infinity;
    const u = ulpGap(a, b);
    if (absd > maxAbs) {
      maxAbs = absd;
      worst = { in: [xv], a, b };
    }
    if (u > maxUlp) maxUlp = u;
  }
  return { op, n, byteExact, maxAbs, maxUlp, worst };
}

function compareBinary(
  op: string,
  produce: (xv: number, yv: number, gv: number) => number,
  ground: (xv: number, yv: number, gv: number) => number,
  useGrad: boolean,
): CompareResult {
  let byteExact = 0,
    n = 0,
    maxAbs = 0,
    maxUlp = 0;
  let worst: CompareResult["worst"];
  const gs = useGrad ? GRAD_UPSTREAM : [1.0];
  for (const [xv, yv] of BINARY_SAMPLES) {
    for (const gv of gs) {
      n++;
      const a = f32(produce(xv, yv, gv));
      const b = f32(ground(xv, yv, gv));
      if (bits(a) === bits(b) || (Number.isNaN(a) && Number.isNaN(b))) byteExact++;
      const absd = Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) : a === b ? 0 : Infinity;
      const u = ulpGap(a, b);
      if (absd > maxAbs) {
        maxAbs = absd;
        worst = { in: [xv, yv, gv], a, b };
      }
      if (u > maxUlp) maxUlp = u;
    }
  }
  return { op, n, byteExact, maxAbs, maxUlp, worst };
}

// ----------------------------------------------------------------------------
// Run the probes
// ----------------------------------------------------------------------------

function fmt(r: CompareResult): string {
  const status = r.byteExact === r.n ? "EXACT " : r.maxUlp <= 2 ? "~1ulp " : "DIVERGE";
  const w = r.worst ? ` worst@[${r.worst.in.map((v) => v.toFixed(3)).join(",")}] ${r.worst.a}≠${r.worst.b}` : "";
  return `  ${status} ${r.op.padEnd(9)} ${r.byteExact}/${r.n} byte-exact  maxAbs=${r.maxAbs.toExponential(2)}  maxUlp=${r.maxUlp}${r.byteExact === r.n ? "" : w}`;
}

function domainFor(op: string): (xv: number) => boolean {
  if (op === "sqrt" || op === "rsqrt" || op === "log") return (xv) => xv > 0;
  return () => true;
}

console.log("=".repeat(78));
console.log("PROBE 3 — REFERENCE: definition-interpreted CPU ref  vs  hand CPU body");
console.log("=".repeat(78));
{
  const results: CompareResult[] = [];
  for (const op of Object.keys(UNARY_DEFS)) {
    const def = UNARY_DEFS[op];
    results.push(
      compareUnary(
        op,
        (xv) => evalExpr(def, { x: xv, y: 0, g: 0 }),
        HAND_UNARY[op],
        domainFor(op),
      ),
    );
  }
  for (const op of Object.keys(BINARY_DEFS)) {
    const def = BINARY_DEFS[op];
    results.push(
      compareBinary(
        op,
        (xv, yv) => evalExpr(def, { x: xv, y: yv, g: 0 }),
        (xv, yv) => HAND_BINARY[op](xv, yv),
        false,
      ),
    );
  }
  results.forEach((r) => console.log(fmt(r)));
  const exact = results.filter((r) => r.byteExact === r.n).length;
  console.log(`  --> ${exact}/${results.length} ops reproduce the CPU body byte-for-byte`);
}

console.log("");
console.log("=".repeat(78));
console.log("PROBE 2 — ADJOINT: derived VJP (adjoint of definition)  vs  table grad");
console.log("=".repeat(78));
{
  const results: CompareResult[] = [];
  for (const op of Object.keys(UNARY_DEFS)) {
    const tableGrad = HAND_UNARY_GRAD[op];
    if (tableGrad === undefined) continue; // op absent from grad checks
    if (tableGrad === null) {
      console.log(`  N/A   ${op.padEnd(9)} table grad = null (non-differentiable); adjoint = ${JSON.stringify(deriv(UNARY_DEFS[op], "x")).slice(0, 40)}`);
      continue;
    }
    const derived = vjpUnary(UNARY_DEFS[op]);
    results.push(
      compareUnary(
        op,
        (xv) => evalExpr(derived, { x: xv, y: 0, g: 1.7 }),
        (xv) => evalExpr(tableGrad, { x: xv, y: 0, g: 1.7 }),
        domainFor(op),
      ),
    );
  }
  for (const op of Object.keys(BINARY_DEFS)) {
    const tg = HAND_BINARY_GRAD[op];
    if (tg === null || tg === undefined) {
      console.log(`  N/A   ${op.padEnd(9)} no ttGrad entry in table (structural)`);
      continue;
    }
    const [dA, dB] = tg;
    const derivedA = vjpBinaryA(BINARY_DEFS[op]);
    const derivedB = vjpBinaryB(BINARY_DEFS[op]);
    results.push(
      compareBinary(
        `${op}.dA`,
        (xv, yv, gv) => evalExpr(derivedA, { x: xv, y: yv, g: gv }),
        (xv, yv, gv) => evalExpr(dA, { x: xv, y: yv, g: gv }),
        true,
      ),
    );
    results.push(
      compareBinary(
        `${op}.dB`,
        (xv, yv, gv) => evalExpr(derivedB, { x: xv, y: yv, g: gv }),
        (xv, yv, gv) => evalExpr(dB, { x: xv, y: yv, g: gv }),
        true,
      ),
    );
  }
  results.forEach((r) => console.log(fmt(r)));
  const exact = results.filter((r) => r.byteExact === r.n).length;
  const near = results.filter((r) => r.byteExact !== r.n && r.maxUlp <= 4).length;
  console.log(`  --> ${exact}/${results.length} derived VJPs byte-match the table; ${near} more agree within 4 ulp (algebraic-rewrite gap)`);
}

// ----------------------------------------------------------------------------
// PROBE 4 — COMPOSITE: RMSNorm as a composition of primitives
// ----------------------------------------------------------------------------

console.log("");
console.log("=".repeat(78));
console.log("PROBE 4 — COMPOSITE: RMSNorm(x) as composition  vs  independent direct impl");
console.log("=".repeat(78));
{
  const eps = 1e-5;
  // Composition of PRIMITIVES: mean-reduction (monoid), mul, add, rsqrt, mul(weight).
  // rms(x) = x * rsqrt(mean(x²) + eps) * w
  function rmsNormComposition(xv: number[], w: number[]): number[] {
    const D = xv.length;
    const sq = xv.map((v) => v * v); // elementwise mul
    const meanSq = sq.reduce((s, v) => s + v, 0) / D; // reduction monoid + scale
    const r = 1.0 / Math.sqrt(meanSq + eps); // add + rsqrt (primitives)
    return xv.map((v, i) => f32(v * r * w[i])); // elementwise mul
  }
  // Independent direct implementation of the mathematical semantics (the reference).
  function rmsNormDirect(xv: number[], w: number[]): number[] {
    const D = xv.length;
    let ss = 0;
    for (const v of xv) ss += v * v;
    const denom = Math.sqrt(ss / D + eps);
    return xv.map((v, i) => f32((v / denom) * w[i]));
  }
  let maxUlp = 0,
    maxAbs = 0,
    exact = 0,
    n = 0;
  const rng = mulberry32(42);
  for (let trial = 0; trial < 200; trial++) {
    const D = 4 + (trial % 60);
    const xv = Array.from({ length: D }, () => (rng() - 0.5) * 8);
    const w = Array.from({ length: D }, () => (rng() - 0.5) * 2);
    const a = rmsNormComposition(xv, w);
    const b = rmsNormDirect(xv, w);
    for (let i = 0; i < D; i++) {
      n++;
      if (bits(a[i]) === bits(b[i])) exact++;
      maxUlp = Math.max(maxUlp, ulpGap(a[i], b[i]));
      maxAbs = Math.max(maxAbs, Math.abs(f32(a[i]) - f32(b[i])));
    }
  }
  console.log(`  RMSNorm  ${exact}/${n} elements byte-exact  maxAbs=${maxAbs.toExponential(2)}  maxUlp=${maxUlp}`);
  console.log(`  (composition uses rsqrt(mean+eps); direct uses 1/sqrt(mean+eps) — the`);
  console.log(`   reciprocal-vs-rsqrt ordering is the only degree of freedom, and it is`);
  console.log(`   exactly the CPU-vs-fused-kernel divergence a real GPU check would probe.)`);
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

console.log("");
console.log("probe complete.");
process.exit(0);
