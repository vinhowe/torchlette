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
