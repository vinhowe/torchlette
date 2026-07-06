# Outer-API sketches: what user code looks like at the end state

Written 2026-07-06 for API review (companion to staged-execution-phase1.md
§6-7, scoped-memory-design.md, and the #66 two-facts contract). Each sketch
is marked [today] / [#70] / [phase-2 proposal]. The concept count for users:
TWO — scope(fn) = memory boundary; capture(fn) = scope + staging. optimizer
.step() is math (PyTorch-shaped); "step" as an engine/memory concept does not
exist in user vocabulary.

## Can capture() fully capture a training step? YES — because the hard parts
were already forced into data by past bugs: GradScaler's inf-check is a GPU
where() (not a CPU branch); optimizer state updates in place with stable
storage (replace-and-hold anti-pattern retired); LR flows as a graph scalar
(SGD-alpha fix). backward() captures because it is deterministic given the
forward.

## Sketches

### 1. Minimal training loop [today]
    for (let i = 0; i < N; i++) {
      const { x, y } = data.next();
      const loss = model.forward(x).crossEntropy(y);
      await loss.backward();
      opt.step();                       // implied boundary; no ceremony
      if (i % 100 === 0) console.log(await loss.item());
    }

### 2. Captured training step [phase-2 proposal]
    const trainStep = api.capture((x, y) => {
      const loss = model.forward(x).crossEntropy(y);
      loss.backward();
      opt.step();
      return loss;                      // declared output
    });
    // loop: trainStep(data.next()); occasional await loss.item() is FREE
    // (outputs always materialized — logging cadence is not a program change)

### 3. Custom optimizer [#70 + in-place contract]
    class Lion extends Optimizer {
      init(param)  { return { m: api.zerosLike(param) }; }  // registered ⇒ persistent
      update(param, grad, { m }, { lr, beta1, beta2 }) {    // hyperparams = 0-d tensors
        const u = m.mul(beta1).add(grad.mul(1 - beta1)).sign();
        m.copy_(m.mul(beta2).add(grad.mul(1 - beta2)));     // in place
        param.sub_(u.mul(lr));                              // in place
      }
    }
    opt.lr = 3e-5;  // property setter WRITES the lr tensor → schedules replay-safe

### 4. Data-dependent control flow (GradScaler, THE seam) [today]
    // WRONG under capture: if (await grads.isfinite().all().item()) opt.step();
    const finite = grads.isfinite().all();
    param.copy_(api.where(finite, newParam, param));        // skip-by-selection
    scale.copy_(api.where(finite, scale.mul(2), scale.mul(0.5)));

### 5. Gradient accumulation [today's semantics; capture composes]
    const micro = api.capture((x, y) => model.forward(x).crossEntropy(y).backward());
    const apply = api.capture(() => { opt.step(); opt.zeroGrad(); });

### 6. Interp: mixed one-off + loop
    const dir = await api.scope(async () => {            // one-off: NO capture
      const pos = model.forward(p, { collectHidden: true }).hidden[L].meanPool();
      const neg = model.forward(n, { collectHidden: true }).hidden[L].meanPool();
      return pos.sub(neg);
    });
    const alphaT = api.keep(api.tensor([0]));            // α as data = warm knob
    const step = api.capture((tok) =>
      model.forwardStep(tok, { hook: (x, l) => l === L ? x.add(dir.mul(alphaT)) : x }));
    slider.oninput = (v) => alphaT.write([v]);           // write, not re-trace
    layerPicker.onchange = () => step.invalidate();      // loud cold re-trace

### 7. Unserved-but-correct: Menagerie mutation (splice layers → captures
    invalidate, runs eager); variable-width beam search (bucket or eager);
    line-search optimizers (predicate or eager). SGD-family all fit.

## The seams (each LOUD, none silent)
1. data→control: tensor-valued branches become where/select, else guard-thrash.
2. scalars: per-step-varying values are tensors/declared slots; closure
   constants freeze; recorder REFUSES undeclared variance.
3. shapes: one tape per bucket.
4. state: survives-the-boundary ⇒ registered + updated IN PLACE (existing contract).
5. effects: outputs are RETURNED; item() inside a captured fn = capture error.
6. RNG: philox offset is in-place state (data) — dropout replays correctly.

## Built-in assumptions
Programs structurally deterministic given structure-inputs; update rules are
tensor-state-in-place (whole SGD family); shape variance bucketable;
repetition worth one recording step. Outside these ⇒ eager fallback, never
breakage.

## Review refinements (2026-07-06, from steelman round)

1. OUTPUT SEMANTICS — "always materialized" needs teeth. The loss TENSOR is
   free (backward root, computed anyway) and the per-step staging COPY is
   ~free, but the handle's validity is a WINDOW: phase-2 obligations are a
   staging RING (K slots), handles valid K steps, LOUD error past the window
   (strict: throw "output from step i awaited after i+K" — never silently
   hand back a later step's bytes). Not awaiting = CPU runahead (the feature;
   training pipelines); runahead BOUNDED (~2-3 steps in flight, JAX-style);
   awaiting every step degenerates to today's serialized behavior (= Fact 2).
   Asymmetry: decode cannot pipeline (sampled token feeds back — per-token
   await is inherent); training can — that's where capture wins biggest.
2. OPTIMIZER SCORECARD — the simple definition gets TODAY: vertical fusion
   (elementwise DAG → ~1 kernel/param via multi-output fusion + in-place fast
   path), scalar/schedule safety (hyperparams-as-data + scalar-table),
   chunking (dispatch-level), GradScaler/clip composition (data). HONEST GAP:
   horizontal packing across params (foreach 100→~8 dispatches) is hand-built
   in Adam only; naive optimizers run ~1 dispatch/param (~1ms GPU floor
   post-tape — usable, not optimal). Closing it = generic "pack same-shape
   elementwise groups" engine pass (the architecture-debt foreach→islands
   stage), which retires Adam's private packing too. The historical "stuff
   you'd never put in PyTorch" was mostly lifetime plumbing — being DELETED
   by #66/#70, not hidden.
3. SHAPE-CHANGE WORRY — empirically bounded: the field converged on shape-
   stability (XLA/TPU largest-ever trainings; padding→sequence-PACKING; MoE
   capacity factors; vLLM paged attention = stable pages). Our failure mode
   is kinder than XLA's (graceful eager fallback + counter vs recompile
   storm) + §7 escape hatches (uniform dims, thrash-demotion). The residual
   worry is STRUCTURE change (mutation/tree compute) — already named in §6
   with its revisit trigger.

4. HORIZONTAL PACKING IS DERIVABLE, NO SPECIFICATION (2026-07-06): PyTorch's
   foreach API is an artifact of EAGERNESS (no program to analyze); lazy =
   whole step program in hand, so discovery is connected-components + the
   existing fused-recipe structure hash. Numerics bit-identical (elementwise-
   only; grad-norm reduction stays its own op) ⇒ ships behind a bit-exact
   differential. Mechanics: level (a) grouped dispatches of ~8 buffers
   (binding limits) — 100 params → ~13 dispatches ≈ hand-built Adam, zero
   layout changes; level (b) 1-2 dispatches needs arena-suballocated params
   + offset table — a deliberate engine-internal layout decision, made
   safely at RECORD TIME. The step-tape is what buys the compiler its
   budget: record-time = the once-per-program moment where expensive passes
   are affordable and repacking is legal (before tapes bind buffers).
   Adam's hand-built foreach = the reference implementation the derived
   pass is differentially tested against, then deleted.
