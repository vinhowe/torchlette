# Staged Execution, Phase 2a: the `capture()` API (decode-shaped)

**Status:** implemented (2026-07-07). Builds ON phase 1 (`docs/staged-execution-phase1.md`;
its §9 records the 2a findings/rules). Training capture is 2b and is NOT attempted here.
**Correctness bar:** inherits phase 1's — silent stale replay is the failure mode;
the G3-class gate is never waived.

## 0. What 2a adds, in one paragraph

Phase 1 shipped a library-owned tape behind hand-rolled driver wiring: the
driver computed an `appKey` string (baking α into it by hand), hand-built the
per-token upload payloads (`buildDecodeUploads`), and drove
`setTapeContext`/`tapeReadyFor`/`tapeReplay` itself. The named caveat: *a
driver that steered while omitting α from its key would silently replay stale
α*. Phase 2a is `api.capture(fn)`: a `CapturedFn` whose coverage is the
**argument list** — derived at call time, no hand-rolled key, no hand-built
uploads — with generateChat migrated onto it and the hand-rolled wiring
deleted.

## 1. The API surface

```ts
const step = api.capture(fn, opts?);   // fn: (...args) => Tensor | Tensor[]
step(...args): Promise<Tensor | Tensor[]>   // always async (replay is async)
step.invalidate(): void                // §6 cold knob: drop skeletons, re-trace
step.stats(): CaptureStats             // calls/hits/traces/coldMisses/…
```

- `fn` is an ordinary inference loop body (noGrad, no optimizer). It runs real
  ops on early calls (the phase-1 recorder observes and derives eligibility);
  once a skeleton is ready and every guard passes, the call replays via the
  tape and the body is short-circuited at its LAST internal upload — before
  any layer compute or in-place op (the §9.1 rule).
- `opts.key?: (...args) => string` — optional STRUCTURAL discriminator for
  structure the arg surface can't express (KV bucket length, cache-instance
  nonce). Not for values.
- `opts.ringDepth?: number` — output staging window (§4), default 3.
- Flag off (`TORCHLETTE_STEP_TAPE` unset): transparent pass-through. capture
  adds ZERO new env flags — it rides the phase-1 campaign flag and its
  TAPE_VERIFY/STRICT_TAPE modes.

## 2. THE ARG-BOUNDARY CONTRACT (value coverage)

Everything that varies must cross the argument list — observable at call time
without executing the body:

| what | knob | mechanism |
|---|---|---|
| TENSOR args | **WARM** — change data every call, zero misses | caller-built pending `tensorFromArray` args: values read synchronously from the un-forced payload and re-dressed onto the skeleton's shape-keyed upload slots (then the arg is DONATED/disposed on a hit); persistent tensor args: external plan inputs whose stable buffer the replay reads live (update in place — the hyperparams-as-data idiom) |
| PLAIN-VALUE args | **COLD-per-value** — counted miss + re-record | hashed into the bucket key; a changed value routes to a new bucket |
| CLOSURE values | **FROZEN at record time** (jax.jit semantics) | nothing re-reads a closure on a hit; documented prominently in the jsdoc, TESTED (`test/capture.spec.ts` "closure values are FROZEN"), `TORCHLETTE_TAPE_VERIFY=N` as the paranoia backstop |
| internal uploads (rope/scatter/mask built inside fn from args+state) | derived | upload interceptor on `tensorFromArray` collects them; a hit throws the short-circuit sentinel at the Nth (learned) upload and reclaims the partial graph by disposing interceptor-tracked wrappers |

Guidance line (in the jsdoc): *"values captured by closure are baked at record
time; pass anything that varies as an argument — a tensor for scrubbed knobs,
a plain value for occasional config."* This maps onto the §6 warm/cold table.

**Why not derive closure-scalar coverage?** The first design (probe the body
for its scalar stream) was empirically unsound — see phase-1 §9.1: a captured
body may not execute past side-effecting ops, and closure scalars (α at layer
14) live past them. The body runs EXACTLY ONCE per call, always for real.

**The engine fix that makes tensor-args warm** (phase-1 §9.3): 1-element
pending `tensorFromArray` is no longer inlined into fused recipes
(`isInlinableScalar`), and the recipe staleness check reads current payloads
regardless of materialization (`inlinedConstantValue`). Before this, an
α-as-tensor flip was silently ignored under default fusion — plain engine, no
tape involved.

## 3. Guards (inherited from phase 1; capture only feeds them)

Derived appKey (arg structure + value-arg hashes + `opts.key`) → bucket;
plan-validity pre-checked at readiness (`stTapeReplayValid`) so a post-commit
decline can't strand in-fn state; boundary regime unchanged; `TAPE_VERIFY=N`
forces every Nth call down the normal path so the executor seam cross-checks
the skeleton (N=1 = full shadow); `STRICT_TAPE=1` throws.

## 4. Output contract — the staging ring

Captured outputs are always materialized (phase-1 harvest). Handles are valid
for a K-call window (ring, default K=3): reading one after K subsequent
captured calls throws a LOUD error naming the step ("output from step i read
after step j (staging ring depth K)") via the tensor-usability seam — never
silently hand back stale bytes. Decode (readback feeds the next token)
degenerates to serial; the ring is the seam 2b's training runahead will use.

## 5. generateChat migration (the reference consumer)

```ts
const decode = api.capture(
  (idx: Tensor) => api.noGrad(() => model.forward(idx, { staticKV, residualHook }).logits),
  { key: () => `kv:bkt${kvBucketLen(staticKV.len + 1, maxSeq)}` },
);
// loop: const logits = await decode(api.tensorFromArray([nextTok], [1, 1]));
```

DELETED: the hand-rolled appKey construction, `buildDecodeUploads` +
`setTapeContext`/`tapeReadyFor`/`tapeReplay` driver wiring (also from
kv-differential's taped arm), and the manual `staticKV.len` advance on hits
(model.forward's own advance was MOVED before the mask upload so the
short-circuit still executes it). The residualHook is closure-captured and
therefore frozen — sound BY CONSTRUCTION here: the CapturedFn's lifetime is
one generateChat call and the hook/α are fixed per generation; a new
generation re-traces with the new hook. The steering demo inherits the tape
path through generateChat.

## 6. What 2b (training capture) needs that 2a does not provide

- Real runahead through the ring (2a decode is serial; K>1 in flight untested).
- backward + optimizer inside the captured region (noGrad-only today): the
  recorder must observe the backward plan + optimizer island as replayable,
  and the in-place optimizer-state contract must hold across replays.
- Data-dependent control flow (GradScaler where-select) validated under capture.
- The §9.1 rule bites harder in training: optimizer in-place updates sit at
  the END of the step, so a training short-circuit point (if any) must be
  chosen so the replay covers ALL in-place state — likely no short-circuit at
  all (the whole step is the tape).
