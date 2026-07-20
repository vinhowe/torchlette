# Unrolled-K Decode: closing the host-in-the-loop so decode joins the whole-step world

*Design + Campaign-1 feasibility probe, 2026-07-19. THE CRYSTAL PUSH, campaign 1
(unrolled-K → functionalization → derive-the-reference). Worktree off `main@1cc28d3e`
(P4b HEAD — "THE SUMMIT DELETION"). Companion to `step-function-compiler-design.md`
(P4a Verdict B / P4b covenant STOP — the blocker this campaign removes) and
`architecture-debt.md`. Design-first: NO mechanism lands here; every number in §2 is
measured on dw-2-1 (A100), distilgpt2, device 10, via `tools/t-uk-*.ts`.*

> Reproduce: `VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim npx tsx tools/t-uk-feedback.ts`
> (probe 1), `t-uk-scale.ts` (probe 2), `t-uk-economics.ts` (probe 3), `t-uk-eos.ts`
> (probe 4). Pretrained arm: `UK_PRETRAINED=1 UK_MODEL_DIR=…/models/distilgpt2`.

---

## 0. One-sentence declaration

> **A K-token decode block is a pure function of (KV cache ∪ parameters ∪ the last
> token ∪ RNG state) → (K sampled ids ∪ advanced KV ∪ advanced RNG), traced once as
> a single static command stream whose per-token sampling feedback (argmax/sample →
> next-token id → next embedding gather) closes ON-DEVICE — so decode stops being a
> host-in-the-loop of K single-token graphs and becomes ONE whole-step-shaped graph
> the same compiler already compiles for training.**

The one-sentence test (§9): *if a decode step cannot be expressed as a node in the
K-block graph or a guard on its inputs — i.e. if it needs a per-token host readback
to decide what runs next — it does not belong inside the compiled block; it is a
block-boundary, by declaration.*

### Why this is the campaign that completes the covenant

P4a's **Verdict B** (`step-function-compiler-design.md`) is the wall: whole-step is a
*training-only* mechanism (its sole lever defers the backward force; decode has no
backward), so it is a structural no-op for decode, and decode keeps running on the
per-plan compile + **step-tape + observed-liveness + cross-plan-edges** substrate.
P4b then rendered the covenant's honest STOP: *every substantial ledger deletion has
a proven live consumer — decode.* The ~2633-SLOC outright ledger (plus ~1100–1900
partial) cannot execute while decode is that consumer.

Unrolled-K removes decode as the consumer **not by extending the backward-deferral
mechanism to decode, but by making decode already single-force and static** — the
exact shape the whole-step build-from-IR compiler + global memory planner consume.
The tape/observed-liveness/cross-plan-edges exist to *approximate at runtime* the
staticness a K-block graph *states by construction*; once decode is a K-block graph,
their last consumer is gone and the P4b ledger executes. **That execution — with a
per-item consumer-absence re-proof for each deleted subsystem — is this campaign's
ultimate acceptance, and it is what takes the covenant net-negative.**

---

## 1. The thesis, stated precisely

Decode today is host-in-the-loop: run one forward → **fence + read the logits back**
→ host argmax/sample → build the next single-token graph → repeat. The readback is
*definitional*: its value (the next token id) is the next graph's input. That single
data dependency is the sole reason decode needs a per-token boundary at all, and the
per-token boundary is the sole reason decode leans on the tape's per-plan re-dress /
observed-liveness convergence / cross-plan witness — the machinery that makes a
*stream of separate small graphs* cheap.

WebGPU forbids device-side control flow, so we cannot loop-until-EOS on the GPU. But
we do not need to: we **unroll**. K tokens become ONE static graph in which the
feedback edge closes on-device — `argmax(logits_t)` produces a token-id *tensor*, and
iteration t+1's embedding **gathers** that tensor directly (the index buffer is read
in-shader). No host sees a token until the K-boundary, where a single readback of K
ids drives streaming + EOS truncation. The block is then exactly a single-force
static graph — the whole-step compiler's native diet.

**Laziness is again the tracer.** Nothing new expresses the block: the same frontend
ops that run eager build the K-block lazily; deferring the single K-boundary force is
the whole acquisition. This is the training arc (`step-function-compiler-design.md`
§1) applied to decode — and it needs *less* than training did, because decode is
already backward-free and single-force; only the per-token readback stood in the way,
and unrolling dissolves it.

---

## 2. FEASIBILITY VERDICT — the four probes (measured, up front)

**Verdict: feasible for greedy TODAY with no net-new mechanism; stochastic sampling
and the compiled-block speed win are the named, bounded remaining work.**

### Probe 1 — GPU-side feedback viability: **PASS (greedy, no host roundtrip).**

`tools/t-uk-feedback.ts`. A real distilgpt2 (pretrained) runs K greedy tokens two
ways over identical weights: **arm HOST** (read logits back each step, host argmax —
today's decode) and **arm DEVICE** (each step's `argmax(logits)` tensor fed straight
into the next step's embedding gather; nothing forced until ONE readback of the K ids
at the block boundary). Prompt "The capital of France is", K=10:

| arm | token stream | submits |
|---|---|---|
| HOST (K readbacks) | `[262, 50256, 464, 198, 198, 198, 198, 198, 198, 198]` | 20 |
| DEVICE (1 readback / block) | `[262, 50256, 464, 198, 198, 198, 198, 198, 198, 198]` | **9** |

**Byte-identical ids** (the stream even emits the EOS id 50256 mid-block), and the
whole K-block forces with a **single** readback vs the host arm's K. The load-bearing
handoff — argmax output → gather index, closed inside one submission — **works today**:

- **argmax outputs an f32 GPU tensor** holding the integer id (`comparison.ts:202`
  returns with no dtype ⇒ f32; the index is a u32 stored as f32,
  `reduction-skeleton.ts:454`).
- **gather/embedding reads f32 indices natively**, coercing `.toU32()` in-shader
  (`ops-tile-ir.ts:740`, `torchlette.ts:1010`) — no cast op, no floor, no host
  roundtrip at the seam.
- **write-then-read is serialized within one submission**: the shared encoder orders
  successive compute passes with WebGPU's implicit inter-pass storage barrier; the
  `sharedEncoderWriteSet` is a pool-reuse (WAR) guard, **not** a RAW-forced flush
  (`buffer-pool.ts:175`, `shared-encoder.ts:338`) — the argmax-write→gather-read edge
  never forces a submit boundary.

**Named sharp-edge (real, precise, non-blocking):** the WebGPU **arg-reduce kernel
assumes a contiguous reduction row** — `argmax` over a *multi-dim strided view*
(`narrow(dim1)` then `narrow(dim2)`) returned the **wrong** index (measured: **40 vs
correct 995**). Materializing the row (`.contiguous()`, cheap — the decode row is one
vector) fixes it and is correct against the host reference. A decode logits row is
naturally contiguous after the lm_head matmul; this only bites the last-position
selection out of a multi-token prefill. Filed as the one seam-assertion the mechanism
must carry (single-source: gather's index dtype/layout derived once, asserted).

**Net-new residue (named, not needed for greedy):** on-device **stochastic sampling**
(multinomial/categorical) does not exist — sampling is host-side `Math.random()` over
a top-K readback (`packages/qwen3-browser/src/generate.ts:93,130`; the GPU `topk` is a
`mapAsync` readback, `topk-kernel.ts:380`). It is *composable* on-device via
**Gumbel-max** = `argmax(logits/temp + gumbel_noise)` using existing Philox `randF32`
(`ops-tile-ir.ts:1133`) + `log` + the argmax above — **no cumsum required** (only a
workgroup-scoped `inclusiveScan` exists; full-vocab inverse-CDF would need a
multi-pass cumsum, which Gumbel-max avoids). See §4.

### Probe 2 — K-block graph scale + compile: **within range; brief's estimate ~4× high.**

`tools/t-uk-scale.ts`. The unrolled-K graph (on-device feedback wiring, distilgpt2,
growing-KV), built and analyzed WITHOUT executing:

| K | nodes N | analyzeGraph ms | nodes/token |
|---|---|---|---|
| 1 | 682 | 5.9 | 682 |
| 2 | 1025 | 5.9 | 513 |
| 4 | 1711 | 8.0 | 428 |
| **8** | **3083** | **17.1** | **385** |

The K=8 block is **3083 nodes / 17 ms** — comfortably inside P0's *measured-linear*
pass range (`step-function-compiler-design.md` §P0: 5859 nodes @ 31 ms, 14589 @ 84 ms).
**The brief's "K=8 ≈ 13k nodes" was ~4× high** — the real block is smaller and the
compile-time headroom larger than assumed. nodes/token *falls* with K (the prefill
amortizes; the growing KV adds a little attention per step). Decode-forward is heavier
per node than training-forward (237 nodes) because the cached path uses the
*decomposed* attention (transpose/matmul/mask/softmax/matmul) not fused SDPA.

**Compile/coverage — the KV regime is the load-bearing distinction.** Two consecutive
unrolled-K=8 blocks (one readback each) grew the template count `0 → 2 → 4`: the
growing-KV block **compiles** (into ~2 build-from-IR templates) but does **not
converge across blocks** — each block's positions are fresh `kvSeqLen` shapes, so
block 1 re-lowers rather than replaying block 0. This is the growing-`cat` KV
(`examples/gpt2/model.ts:164`), and it is exactly why **the design mandates the
static-KV path** (`packages/qwen3|gemma2 forwardStatic`): there position/token/scatter/
rope enter as **DATA** index-tensors and only the 128-token bucket length is shape, so
K in-bucket steps **share ONE template** (P4a measured decode steady template-growth
0). On that path an unrolled-K block is one recurring template that compiles once and
**replays**. (The exact build-from-IR *uncovered label set* — the CE strided-narrow
bail + `op:max` residue named in `step-function-compiler-design.md` — could not be
captured here: the `generateStream` export is a frozen ESM binding and the branch's
`DIAG_CUTOVER` dump is not in this tree. Capturing it on the static-KV path is P1's
first census, §5.)

### Probe 3 — the economics: **host tax is ~98% of today's per-token wall.**

`tools/t-uk-economics.ts`, distilgpt2 growing-KV decode, 32 steps, late-half medians:

| arm | build | readback | hostarg | markStep | **TOTAL** |
|---|---|---|---|---|---|
| tape OFF | 1.55 | 39.60 | 1.39 | 12.50 | **54.76 ms/tok** |
| tape ON | 1.48 | 39.43 | 1.40 | 12.24 | **54.94 ms/tok** |

Tokens byte-identical OFF vs ON; **the tape adds nothing measurable to decode**
(consistent with the P4a decode-α premise). The per-token **host tax**
(readback + hostarg + markStep = **53.5 ms**) is **~98%** of the 54.76 ms wall — the
`await logits.cpu()` **fence** (39.6 ms) dominates, exactly the host-in-the-loop cost
unrolled-K pays once-per-K instead of once-per-token. The naive host-tax-amortization
projection (÷K) reaches 8× at K=8 / 16× at K=16 — but that is an **upper bound**;
Probe 4 measures the floor it ignores.

### Probe 4 — EOS predication cost + the honest per-iteration GPU floor.

`tools/t-uk-eos.ts` scales the unrolled block (ONE readback per block, so K iterations'
GPU compute is the wall minus a single fence). Least-squares fit over K∈{2,4,8,16}:

| K | block wall ms | ms/token |
|---|---|---|
| 2 | 108 | 54.0 |
| 4 | 162 | 40.5 |
| 8 | 278 | 34.8 |
| 16 | 536 | 33.5 |

**`block_wall ≈ 40.4 ms (once-per-block host tax) + 30.8 ms/iter × K`.** The slope —
**30.8 ms/iter** — is the per-iteration GPU-forward cost in the **LOWERED (uncompiled)**
unrolled regime, and it is the honest correction to Probe 3: the per-token floor is
**not** ~0. So on this A100, **uncompiled** unrolled-K goes **54.76 → 33.5 ms/tok at
K=16 — a ~1.6× win** (host-tax amortization only), NOT 16×. The full multiplier is
gated on two things this campaign delivers/targets: **(a)** the block COMPILING and
replaying (Probe 2: it compiles; the static-KV path replays — the compiled forward is
far cheaper than the 30.8 ms lowered one), and **(b)** the **browser** — the real
target (per the brief) — where the per-token host tax (JS dispatch + submit + a slower
readback bus) is materially larger, so the amortization win is proportionally bigger.

**EOS.** Because the stream is static, an EOS at token j<K cannot break it; the
remaining K−j iterations execute and the host truncates at the K-boundary readback.
**Unpredicated** waste = the slope × wasted iters, and only the *final* EOS-containing
block wastes (every full block emits all K): expected ≤ **(K−1)/2 forwards, one-time**
at end-of-generation (K=8 ≈ 108 ms once; negligible amortized over T≫K tokens).
**Predication** (a flag-buffer early-return kernel) would drop each wasted iteration
from a full forward to ~kernel-launch overhead — but since the unpredicated waste is
already small vs the host tax and one-time, **predication is a second-order
optimization; the simple design (compute all K, truncate at readback) is viable
without it.** Named as an optional refinement, not a v1 requirement.

**Zero genuine blocker for greedy unrolled-K.** The feedback primitive exists, the
graph scale is in range, the economics favor it (more so compiled / in-browser), and
EOS is handled by truncation. The only net-new work is stochastic sampling-as-ops
(§4) and the compiled-replay maturation the static-KV path already implies.

---

## 3. Architecture

### 3.1 The K-block as a whole-step-shaped function

An unrolled-K decode block is `S_dec`, defined by:

- **Inputs** = the static KV cache buffers (persistent, `registerState`), the
  parameter set, the last token id (a leaf), the current bucket length (a *shape*
  constant, §3.4), and the RNG state at block entry (as DATA, §4).
- **Outputs** = K sampled ids (one readback at the boundary), the in-place-advanced KV
  (scatter at positions t+1..t+K), and the advanced RNG counter.
- **Everything internal** — per-iteration logits, attention temporaries, the argmax/
  sample id tensors feeding the next gather — is owned by the block's memory plan and
  is never observable; reading any of it mid-block is a block-boundary break (§9).

This is the whole-step contract (`step-function-compiler-design.md` §0) with
"backward" replaced by "the K-1 internal feedback edges." Crucially, `S_dec` needs
**no backward-force deferral** — it is *already* single-force. The whole-step compiler
consumes it because it is a static single-force graph, which is all that compiler ever
required; the training-specific deferral was only ever how *training* reached that
shape. Decode reaches it by unrolling.

### 3.2 Sampling as declared ops (the feedback edge)

The feedback edge is three existing ops per iteration, all on-device (Probe 1):
`logits_t → [temp/sample transform] → argmax → id tensor [1,1] → reshape → embedding
gather (iteration t+1)`. Greedy is `argmax` alone. Stochastic sampling adds a declared
prologue (§4). The id tensor is the next iteration's gather index directly — **no
separate "write the id into an index buffer" op is needed** for greedy (the argmax
output *is* the index buffer); a sampled id is likewise a tensor the gather reads.

### 3.3 The streaming / EOS contract — K-granularity, host truncation

- **Streaming granularity is K.** The UI receives tokens in blocks of K (one readback
  per block). This is the deliberate trade: coarser streaming for host-loop-free
  decode. K is a tunable knob (born with a sunset per house policy, §5).
- **EOS by host truncation at the K-boundary.** The K ids come back together; the host
  finds the first EOS and truncates the block there, then stops. Post-EOS iterations
  in that final block are computed-and-discarded (Probe 4: one-time, ≤(K−1)/2 forwards).
- **Optional EOS predication** (flag-buffer early-return kernels) is a second-order
  refinement (Probe 4), not v1.

### 3.4 Bucket-boundary handling — K clipped to the bucket edge

On the static-KV path the graph shape is constant *within* a 128-token bucket
(`KV_BUCKET`, `packages/qwen3-browser/src/model.ts:93`) and changes only at bucket
boundaries. **A K-block is clipped to the bucket edge**: if positions t+1..t+K would
cross a 128-boundary, the block is shortened to end at the boundary; the next block
starts in the new bucket (a new — but still recurring — template). This keeps every
block a single stable template that compiles once and replays, and confines re-lowering
to the O(seq/128) bucket transitions. `seqLen===1` per iteration is required by the
current `forwardStatic` mask path (`model.ts:612`); the K iterations are K single-token
steps sharing the template, chained by the on-device feedback — NOT one K-wide
attention (which would need lifting that restriction; out of scope for v1).

### 3.5 Temperature / top-p as DATA via GPU RNG

RNG today is host-seeded (`_rngCounter → u32 uniform`, `engine.ts:412`), not a
graph-threaded GPU state tensor. For a *replayed* static block to advance RNG
deterministically, the per-block seed/counter must flow in as **DATA** (an uploaded
uniform/tensor) — exactly the scalars-as-data pattern the whole-step campaign already
established for Adam's step_size / GradScaler's inv_scale / scheduled LR
(`architecture-debt.md`). Temperature and top-p thresholds are likewise data. On-device
categorical is **Gumbel-max**: `id = argmax(logits/temperature + (-log(-log(U))))`,
`U = randF32(seed_as_data, offset)` — composed from existing exp/log/argmax/Philox,
no cumsum. Top-k as a *pre-filter* over a small K (≤64) fits a single-workgroup
`inclusiveScan` (`tile-ir.ts:2984`); full-vocab inverse-CDF (multi-pass cumsum) is the
one sampling scheme that stays host-side (§8).

### 3.6 What the compiler does with `S_dec` — nothing new

`S_dec` goes through the *same* passes as a whole-step training graph (P0-scaled,
Probe 2 in-range): CSE/DCE, fusion, the memory planner over the block's static
liveness. On the static-KV path it reaches build-from-IR and replays under an input
guard (bucket length + last-token shape + structural fingerprint) — a strict subset of
the tape's guard taxonomy, reused then trimmed. Decode thereby stops needing the
tape's per-plan re-dress and observed-liveness convergence: the block *states* its
liveness; the runtime learner has nothing left to discover.

---

## 4. Sampling: what moves on-device, what stays host (the usage boundary)

| Scheme | On-device? | How | Residue |
|---|---|---|---|
| **Greedy (argmax)** | **Yes, today** | Probe 1 | none |
| **Temperature** | Yes | scale logits by `1/temp` (data) before argmax/Gumbel | none |
| **Categorical / multinomial** | Yes (net-new op) | **Gumbel-max** = argmax(logits/temp + gumbel); Philox `randF32` + log + argmax | seed-as-data (§3.5) |
| **Top-k (small k≤64) sampling** | Yes (net-new) | workgroup `inclusiveScan` prefilter + Gumbel over the k | none |
| **Top-p / nucleus (full vocab)** | **No — stays host** | needs a sorted full-vocab CDF (multi-pass cumsum over 50k–256k) | **the typed residue** |
| **Arbitrary host sampler** (custom logits processors, grammar-constrained, beam) | **No — block-boundary** | reads logits mid-stream ⇒ host-in-the-loop by definition | falls back to K=1 (today's decode) |

**The typed usage boundary:** any sampler that must **read the full logits
distribution on the host mid-stream** — full-vocab top-p, grammar/logit-bias
processors that mutate logits per token, beam search — **refuses to move on-device**
and runs at **K=1** (today's per-token decode, which survives as the correct fallback).
This is the decode analogue of the whole-step "mid-step poke ⇒ eager for that step"
rule (`step-function-compiler-design.md` §6): a mid-block host read is a block-boundary,
by declaration. Greedy + temperature + on-device categorical + small-top-k cover the
shipped demo paths (qwen3-steering / gemma); the residue is named, bounded, and
non-blocking.

---

## 5. Phased campaign plan (shippable, differential-first)

The mother gate governs every phase: **the unrolled-K block must byte-match the
per-token host-loop reference** — greedy ids byte-identical (Probe 1 is this gate at
K≤16); sampled streams identical under a fixed seed-as-data. No phase lands without it
green, both directions of every flag. Every phase names its deletions (house policy);
every new flag is born with a sunset.

- **P0 — Probes + design (THIS PASS, landed).** The four probe verdicts (§2), the
  design (this doc), and the named sharp-edges (strided-argmax contiguity; RNG-as-data;
  bucket clipping). Ships nothing user-visible. *Done.*

- **P1 — The static-KV unrolled-K block behind a flag (`TORCHLETTE_UNROLLED_K`).**
  Build `S_dec` on the `forwardStatic` path (qwen3/gemma2): K in-bucket steps chained
  by on-device argmax→gather, one K-boundary readback, bucket-clipping. Greedy only.
  Gate: greedy stream byte-identical to K=1 host-loop over ≥2 buckets; **the
  build-from-IR uncovered census** on the static-KV block (the label set Probe 2 could
  not capture here — expect it EMPTY or the named CE/`op:max` residue, and close any
  bail like P2 closed the CE strided-view bail). Carries the strided-argmax contiguity
  assertion at the gather seam. *Productizes Probe 1 on the compile-eligible path.*
  **LANDED 2026-07-19** — see "P1 STATUS" below. The census came back **{ arg-reduce }**
  (not CE/`op:max`): the block's `argmax` feedback op is the single build-from-IR
  uncovered op, and it is P2's precise coverage input.

- **P2 — Compile + replay the block; measure the real multiplier. PARTIAL — see
  "P2 STATUS" below.** Route `S_dec` through the whole-step build-from-IR compiler.
  **Landed:** three single-sourced generators (arg-reduce, max/min routing,
  fusedRMSNormForward) that shrink the block's runtime census from a *reasoned*
  `{arg-reduce}` (P1, optimistic) to a *measured* **`{fusedRoPE}`** — the block is now
  ONE generator from full coverage. **Measured multiplier:** the host-tax amortization
  is **~1.8× (host/block-low, N=64 A100)**, re-confirmed at scale. **Not yet realized:**
  compiled *forward* replay — `fusedRoPE`'s per-position offset is a per-replay-varying
  uniform that must flow as DATA (a volatile config repack), so today build-from-IR is
  net-negative on the block (wasted failed-coverage attempts) and decode runs best
  lowered. *The compiled win the design projected is gated on the bounded RoPE-offset-
  as-data follow-on, NOT on arg-reduce.*

- **P3 — Stochastic sampling on-device (Gumbel-max + seed-as-data).** Add the
  categorical prologue (§3.5) and small-top-k; thread the per-block seed as data.
  Gate: sampled stream identical to a host Gumbel-max reference under a fixed seed;
  RNG advances deterministically across replayed blocks. Names the full-vocab top-p /
  arbitrary-host-sampler residue as the typed K=1 fallback (§4). *Covers the shipped
  demo samplers on-device.*

- **P4 — DEMO-PATH CUTOVER (cutover discipline — the live consumers).** Move
  `examples/qwen3-steering` and the gemma SAE demo decode paths onto unrolled-K,
  behind the flag, then default-on, then remove the K=1-per-token host loop for the
  covered samplers. Gate: the demos' token streams byte-identical pre/post cutover;
  the browser suite green; the SAE-steering interactivity unregressed. **These are the
  live consumers `step-tape*.ts` was retained for — the cutover is what frees them.**

- **P5 — THE LEDGER EXECUTION (the ultimate acceptance).** With decode proven no
  longer a consumer of the tape / observed-liveness / cross-plan-edges — *per-item,
  each with a consumer-ABSENCE re-proof mirroring the P4b presence-proofs* — execute
  the P4b deletion ledger subsystem by subsystem, each behind a green parity gate:
  - **Re-run `tools/t-p4b-decode-edges.ts` under unrolled-K** ⇒ expect
    `crossPlanEdgeStats().producers = 0` and no observed-liveness convergence on the
    decode path (the P4b *presence* proof was `producers=1, convergedTemplates=1`; the
    deletion gate is that both go to 0 because the block states its liveness).
  - **Re-run `tools/t-decode-whole-step.ts`** ⇒ expect **Verdict A** (decode now traces
    + compiles as a distinct static function), inverting P4a's Verdict B.
  - Then delete, in order, with each file's decode+training consumer-absence proved:
    `cross-plan-edges.ts` (152) → `tape-profile.ts` (18) → `step-object.ts` (156) →
    `step-tape-replay.ts` (680) → `step-tape.ts` (820) → `observed-liveness.ts` (807),
    plus the partial harvest/arena/pool reductions (~1100–1900). The retained
    two-plan-eager path + `CHECKPOINT_EAGER_REFUSAL` (P4b PERMANENT POLICY) are a
    *training* concern, unaffected — this ledger is the decode+runtime-staticness half.

- **P6 — Covenant reckoning + sunsets.** Weight-norm the deletion; sunset
  `TORCHLETTE_UNROLLED_K` and the K knob once default-on and the K=1 host loop is gone
  for covered samplers.

**Projected covenant numbers** (code-only SLOC; current `src` = **66066** at P4b HEAD;
pre-Everest baseline **65795**, currently **+271** — net-positive, the covenant's open
debt):

| item | ΔSLOC |
|---|---|
| unrolled-K mechanism (feedback declaration, Gumbel-max + small-top-k ops, seed-as-data, bucket-clipping, optional EOS predication) | **+~300–600** (est.) |
| ledger — deleted outright (obs-liveness 807 + step-tape 820 + replay 680 + step-object 156 + cross-plan-edges 152 + tape-profile 18) | **−2633** |
| ledger — deleted partial (harvest/witness seams, arena/pool runtime-adaptive, planner epoch churn) | **−1100 … −1900** |
| **campaign net vs current 66066** | **−3400 … −3900** |
| **projected final `src`** | **~62200 … ~62700** |
| **vs pre-Everest 65795** | **−3100 … −3600 (NET-NEGATIVE)** |

The covenant completes: adding the small, declared unrolled-K mechanism and executing
the ledger it unblocks takes the campaign **decisively net-negative** — the largest
deletion in project history landing because decode finally joined the whole-step world.

---

## 6. Risks (honest)

- **The demo paths are live consumers — cutover discipline is load-bearing.**
  `examples/qwen3-steering` sets `TORCHLETTE_STEP_TAPE=1` unconditionally
  (`src/lib/tape-flag.ts`) and the gemma SAE demo decodes live. Deleting the tape
  BEFORE these are cut over to unrolled-K (P4) breaks the shipped product. **P5 (delete)
  is gated on P4 (cutover) with byte-identical stream proofs.** No ledger file is
  deleted while any decode demo still routes through it — the P4b STOP rule, honored.
- **Static-KV `seqLen===1` restriction.** `forwardStatic` throws on multi-token decode
  (`model.ts:612`); v1 is K single-token steps sharing a template, not one K-wide
  attention. If a future workload needs K-wide, that restriction must lift — named,
  out of scope.
- **RNG determinism under replay.** Seed-as-data (§3.5) is essential and unbuilt; a
  replayed block that re-seeds from the host counter would diverge run-to-run. Gate:
  fixed-seed sampled-stream identity across replayed blocks (P3).
- **Dawn teardown segfault class.** Every unrolled probe here core-dumped on exit
  AFTER printing correct results — the known Dawn background-thread teardown segfault
  (`process.exit(0)` is called; results are emitted first). Harmless for probes; but
  any long-running decode server must not treat a teardown crash as a decode failure.
  Named so it is not re-misattributed.
- **The full speed multiplier is a projection, not yet measured.** Probe 4 gives ~1.6×
  uncompiled on A100; the compiled + in-browser win is P2's to prove, not assumed here.
- **Strided-argmax is a real correctness edge.** The arg-reduce-over-strided-view bug
  (Probe 1: 40 vs 995) must be carried as a seam assertion (contiguity of the reduction
  row), or a future non-contiguous logits layout silently mis-samples.

---

## 7. Red-team

**Objection 1 — "You are trading correctness/latency for a SLOC number: K-granularity
streaming is worse UX, and computing post-EOS tokens is wasted work, all to delete
files."** *Ruling: rejected on the measurement.* The streaming coarsens to K tokens
per readback, but the *decode itself* gets ~1.6× faster uncompiled (Probe 4) and more
compiled/in-browser (P2) precisely because the per-token fence (39.6 ms, 98% of the
wall — Probe 3) is amortized. Post-EOS waste is one-time, ≤(K−1)/2 forwards (Probe 4),
negligible for real generations. The deletion is the *consequence* of decode becoming
static, not the goal contorting the design — the goal is host-loop-free decode, which
is independently a speed + architecture win, and the ledger falls out of it.

**Objection 2 — "The whole-step compiler is a TRAINING mechanism (Verdict B says so);
you cannot bolt decode onto it."** *Ruling: this is the sharpest objection; it rests
on conflating the *mechanism* with the *shape*.* Verdict B is precise: whole-step's
*trace-acquisition lever* (deferring the backward force) is training-only and inapplicable
to a backward-free decode. But the thing that lever *produces* — a static single-force
graph consumed by build-from-IR + the global memory planner — is shape, not mechanism.
Unrolled-K reaches that shape by a *different* acquisition (unrolling the host loop
into on-device feedback), then feeds the *same* compiler. Probe 2 shows the block
compiles through the existing passes at in-range scale. Decode does not "bolt onto"
whole-step; it independently arrives at the same static graph the compiler already eats.

**Objection 3 — "Deleting the tape/observed-liveness will silently break the shipped
decode demos — the exact P4b STOP."** *Ruling: conceded as the real hazard; addressed
by phase ordering + consumer-absence re-proofs.* The P4b STOP was honorable *because*
decode was a proven live consumer (`producers=1`). This campaign does not force-delete;
it **removes the consumer first** (P4 cutover, byte-identical stream gate) and only
then deletes, with a **per-item consumer-ABSENCE re-proof** (re-run the very P4b probes
and require `producers=0` / Verdict A). A file is deleted only when its decode
consumer is *proven gone*, not assumed — the STOP rule's own logic, run forward.

---

## 8. Open questions (only those that materially fork the design)

1. **Is full-vocab top-p worth an on-device multi-pass cumsum, or is K=1 fallback
   acceptable forever?** This forks §4's residue: if the shipped product needs
   full-vocab nucleus sampling *at unrolled-K speed*, a multi-workgroup cumsum
   (net-new, ~a kernel) enters src; if K=1 fallback for that sampler is acceptable
   (greedy + temperature + categorical + small-top-k cover the demos today), the
   residue stays host-side and no cumsum is built. **For Vin:** do the steering demos
   need full-vocab top-p, or is small-top-k / temperature sufficient? (Materially forks
   whether an on-device cumsum lands.)

2. **K default + streaming-latency policy.** K trades streaming granularity for
   host-loop amortization. What is the default K, and is it fixed or adaptive
   (small K near EOS-likely regions, large K in bulk)? A policy knob, not an
   architecture fork — but it should be declared, not discovered. (Ask only if the UX
   target constrains it.)

Everything else (the static-KV substrate, the feedback primitive, the phase order,
the ledger) is determined by the design + measurements and needs no ruling.

---

## 9. One-sentence test

*If a decode step needs a per-token host readback to decide what runs next — a sampler
that reads the full distribution on the host mid-stream — it is a block-boundary by
declaration and runs at K=1; everything else is a node in the K-block graph or a guard
on its inputs, and the block is the compiled function that lets decode join the
whole-step world and the P4b ledger execute.*

---

## P1 STATUS — the static-KV unrolled-K block, landed 2026-07-19

*Worktree off `main@559c99eb`. Staged commits, no push. dw-2-1 (A100), random-init
Qwen3, node v22 + Dawn/Vulkan vk-shim. Reproduce the gate:
`eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX
LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-block-diff.ts`.*

### The sharp edge, fixed on the main path (unconditional)

The arg-reduce-over-strided-view bug (§2 Probe 1: **40 vs 995**) was a standing
framework correctness bug: the WebGPU arg-reduce kernel derives its input addressing
from `contiguousStrides(inputShape)` and binds the buffer flat from element 0, so a
narrow **offset** view (a last-position row) or a multi-dim **strided** view was
indexed from the wrong base with the wrong strides — silently returning the wrong
index. **Fix** (`src/backend/webgpu/ops/comparison.ts`): `argReduceOp` materializes the
input contiguous (offset 0) when `!isContiguous || offset!=0`, mirroring the sum/max
reduction guard (`reductions.ts`, task #58), plus a **seam assertion** that the kernel's
contiguous-stride assumption holds (single source of truth at the seam). Cheap — a
decode reduction row is one vector. **Failing-first spec:** `test/argmax-strided-view.spec.ts`
(webgpu) — pre-fix returns row-0's index (2 vs 7, 1 vs 3); passes with the guard. Landed
independent of unrolled-K.

### The surface as landed (`packages/qwen3-browser/src/generate.ts`)

Proven in a package before `src/` (admission pressure). Greedy only (P3 adds sampling).
- **`decodeBlock(api, model, kv, lastTok, K, {residualHook, stopTokens})`** → `{ ids, stopIndex }`.
  K greedy steps as ONE lazy graph: each step's `argmax(logits)` id-tensor feeds the next
  step's embedding gather **on-device**; the KV scatters at static positions (advanced
  host-side at graph BUILD time → per-step rope/scatter/mask carry positions as DATA);
  **one** readback of the K ids at the boundary; host truncation on the first stop token
  (compute-all-K, truncate-at-readback — §3.3).
- **`clipBlockToBucket(len, K, maxSeq)`** — clips K to the 128-bucket edge / maxSeq so
  every block is a single stable template (§3.4).
- **`unrolledKFromEnv()`** — browser-safe read of `TORCHLETTE_UNROLLED_K`.
- **`StaticDecodeModel`** — the minimal `forward({staticKV})` interface; Qwen3 and Gemma2
  both satisfy it (gemma2 wiring is a mechanical port, deferred to P4 cutover).
- **`generateChat` wiring:** greedy (`temperature===0`) **and** `TORCHLETTE_UNROLLED_K>=K`
  decodes in K-blocks; **every** other case (all sampling, flag off) takes the per-token
  host loop **unchanged** — the block branch is compile-time-guarded, so decode paths are
  byte-identical when the flag is off.

### The flag's sunset

`TORCHLETTE_UNROLLED_K` (integer K; unset/<2 = off) is born 2026-07-19 with a sunset
(house policy): **soak** (opt-in) → **default-on** (P4 cutover, byte-identical stream
gate) → the K=1 host loop is removed for greedy and the flag/knob **dies** (P6). A flag
outliving P6 is debt.

### The mother gate — the differential (`tools/t-uk-block-diff.ts`)

Block ids **byte-identical** to the per-token host-loop reference across **3 prompts ×
K∈{1,4,8,16} × a 128-bucket crossing** (all PASS); the EOS contract (compute-all-K, the
`stopIndex` is the first-occurrence of the stop, host truncation keeps exactly the
pre-stop tokens, no-stop ⇒ no truncation); and the bucket-clip unit checks. Economics
(N=16, K=8, **uncompiled/lowered**):

| arm | submits | ms/tok |
|---|---|---|
| host per-token loop | 48 | ~100 |
| **unrolled block** | **15** | **~37** |

**~3.2× fewer submits, ~2.7× faster ms/tok** — the ~1.6×-class host-tax amortization,
measured honestly. The full multiplier is P2's (the block compiling + replaying) and the
browser's (a larger per-token host tax). *(A natural mid-block EOS is unreachable here —
random-init greedy collapses to a constant token; Probe 1's real pretrained distilgpt2
emitted EOS mid-stream, and the truncation **contract** is gated above against the actual
static-KV stream regardless of degeneracy.)*

### The build-from-IR uncovered census (P1's first census → P2's input)

Grounded in the generator table (`src/executor/stream-generate.ts` `generateSequential`),
which routes **sum/mean** reductions, elementwise, **gather**, **scatterAdd**, **cat**,
layernorm, and cross-entropy — and hits `miss(label)` (→ uncovered, keeps record/replay)
for any op with no generator. The base static-KV decode forward already compiles and
replays (Probe 2 / `t-decode-template-count`: steady template-growth 0). The unrolled
block adds exactly **one new-to-decode uncovered op:**

| op the block adds | build-from-IR generator? | consequence |
|---|---|---|
| `argmax` (arg-reduce — the feedback selection) | **NO** (`generateSequential` has no arg-reduce case; only sum/mean) | the block plan is not `fullyCovered` → stays record/replay (lowered) |
| `reshape` (feedback view) | n/a (metadata view, no action) | — |
| `cat` (K-boundary id concat) | yes (`generateCat`) | covered |
| embedding `gather` (on-device feedback) | yes (`generateGather`) | covered |

So the census is **{ arg-reduce (`argmax`/`argmin`) }** — NOT the CE/`op:max` residue the
design hedged for (CE is training-only; softmax's max-subtract is inside the fused decode
kernel, already covered). This is **correct and expected for P1** (P1 is the lowered
block; §5 P1 explicitly ships uncompiled). **P2's coverage input is precise:** add an
arg-reduce build-from-IR generator mirroring `serializeReduction` (the sum/mean
single-source realizer already exists in `schedule/reduction-skeleton.ts` — `argReduceWGSL`
is the byte-differential kernel; P2 wires its command-stream generator). Closing that one
op is what lets the block reach compiled replay (P2's gate).

### Gates run

`tools/gate-wall.sh --profile training` (green); `test/argmax-strided-view.spec.ts` +
`tools/t-uk-block-diff.ts` (the differential, green); `tools/t-uk-feedback.ts` re-run
green; decode paths unchanged when the flag is off (block branch compile-time-guarded);
strict-lifetime default. Weight-norm: `src` SLOC unchanged for the product surface (it
lives in `packages/`); the argmax fix adds ~20 code lines to `src/backend/webgpu/ops/comparison.ts`.

---

## P2 STATUS — compiling the block: generators, the corrected census, the real multiplier (2026-07-19)

*Worktree off `main@4334bf9d` (P1 HEAD). Staged commits, no push. dw-2-1 (A100),
random-init Qwen3 (packages/qwen3-browser, the static-KV `forwardStatic` path),
node v22 + Dawn/Vulkan vk-shim. Reproduce: `eval "$(tools/pick-gpu.sh)";
LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-multiplier.ts`
(the multiplier), `… tools/t-uk-generators-parity.ts` (generator compiled-parity),
`… tools/t-uk-block-diff.ts` (the mother gate + compiled arm).*

### The corrected census — P1's `{arg-reduce}` was optimistic; the truth is `{fusedRoPE}`

P1's census was a *reasoned* claim from the generator table, not a runtime capture,
and it was wrong: it assumed "the base static-KV decode forward already compiles"
and so named `arg-reduce` as the block's ONLY uncovered op. The **runtime** census
(`TORCHLETTE_DEBUG_CENSUS=1`, gated at the build-from-IR `generateStream` seam) on
the real K=8 block — a single **~1049-node** plan (the whole unrolled block, one
force) — shows the decode forward leaves **five** classes uncovered on first sight
and, after warmup (config buffers populated), **three** that persist:

| uncovered op (steady) | per-step | why | P2 action |
|---|---|---|---|
| `argmax` (arg-reduce feedback) | 1 | `generateSequential` had no arg-reduce case | **generator added** ✓ |
| `op:max` (softmax max-subtract) | 2 | routing only sent `sum`/`mean` to the monoid-generic serializer | **routed max/min** ✓ |
| `fusedRMSNormForward` | 9 | no generator (Qwen3 is RMSNorm, not LayerNorm) | **generator added** ✓ |
| `fusedRoPE` | 4 | **per-position offset is a per-replay-varying uniform** | **the lone residual — see below** |

(`op:sum[config-missing]` and `fused[no-input-pattern]` are FIRST-EXECUTION artifacts
— the tile plan's config buffer / external-input pattern is captured by
`populateCapturesFromIR` on the first run and covered on the 2nd+; they are NOT
steady uncovered ops.) After the three P2 generators, coverage on the block is
**972/1004 actions**, uncovered = **`{fusedRoPE}` only**.

### What landed — three build-from-IR generators, all single-sourced

1. **arg-reduce** (`generateArgReduce`, +`planArgReduceDispatch` single-sourced with
   `argReduceOp`): ALLOC(indices) + one dispatch `[input, output, params(outSize,
   dimSize, dimStride)]`, WGSL from the shared `argReduceWGSL`. Inherits the P1
   contiguity seam (`argmax`/`argmin` → `[0]` in `CONTIGUOUS_OPERANDS`).
2. **max/min reduction routing**: extended `generateSequential`'s reduction guard from
   `sum|mean` to `sum|mean|max|min` — the serializer was already monoid-generic
   (`planFull/DimReductionDispatch` take `decl.monoid`); max/min differ only in the
   declared monoid. No new codegen.
3. **fusedRMSNormForward** (`generateRMSNormForward`, +`planRMSNormForwardDispatch`
   single-sourced with `dispatchRMSNormForward` via the shared `rmsNormFwdSpec`):
   structurally identical to the proven `generateLayerNormForward` (two inputs, no
   bias). `fusedRMSNormForward` → `[0,1]` contiguity-required.

**Single-source argument:** every generator consumes the SAME plan helper / kernel
spec the imperative dispatcher uses (the `planGatherDirect` pattern), so a divergence
is a build-time DIVERGE, never a silent wrong stream. **Guardian:**
`tools/t-uk-generators-parity.ts` builds a fully-covering graph
`argmax(exp(rmsnorm(x,w) − max(rmsnorm(x,w))))`, runs it past the cutover threshold,
and asserts the COMPILED ids are byte-identical to the lowered path AND that a
template containing all three ops reached compiled replay (CLAUDE.md Corollary 2 — the
differential crosses the activation threshold). **PASS.**

### The one remaining blocker — `fusedRoPE`'s per-position offset

RoPE's cos/sin are **narrow row-slices of a persistent `[maxSeqLen, D/2]` table**;
the fused kernel folds the view's **element offset** (= position × D/2) into its
config uniform (`model.ts` "folds the view's element offset", `rope-kernel.ts`
`cos_offset`/`sin_offset`). That offset is **per-position**, so across the K unrolled
steps AND across replayed blocks it VARIES. A build-from-IR generator that bakes the
record-time offset would replay every block at the wrong position → wrong logits (the
frozen-uniform class CLAUDE.md warns silently corrupted training twice). Covering RoPE
correctly therefore requires the offset to flow as **DATA — a volatile `TAG_UNIFORM`
that re-derives `cos_offset`/`sin_offset` from the current node per replay**
(`directInputOffset`/`deriveNodeOffset` already exist; the missing piece is a
volatile-config repack for the RoPE tile-kernel, analogous to `setAdamConfigUniforms`).
That is a bounded, named follow-on — NOT a new-op generator — and it is the single
gate between the block and full compiled replay.

### THE REAL MULTIPLIER (measured, honest — random-init Qwen3 8L/512d, N=64, 3 repeats, medians)

| K | block-low ms/tok | subs | block-def ms/tok | subs | host/low | host/def | low/def |
|---|---|---|---|---|---|---|---|
| 4 | 59.40 | 144 | 73.84 | 144 | **1.72×** | 1.39× | 0.80× |
| 8 | 57.42 | 120 | 75.16 | 120 | **1.78×** | 1.36× | 0.76× |
| 16 | 57.00 | 112 | 74.21 | 112 | **1.80×** | 1.38× | 0.77× |

*host per-token loop: 102.31 ms/tok, 192 submits (K-independent).*

- **block-low** = the unrolled K-block with build-from-IR DISABLED
  (`TORCHLETTE_COMPILED_PLAN=0`) — pure lowered replay. **host/low ≈ 1.8×**: the
  host-tax amortization (one fence per K instead of per token), ~1.7× fewer submits.
  **This is the measured decode win today** — the same class P1 saw (~2.7× ms/tok on
  the small config), re-confirmed at scale.
- **block-def** = build-from-IR ENABLED (default) is **SLOWER** (low/def ≈ 0.77×):
  because the block does **not** reach compiled replay (RoPE uncovered), every block
  wastes a failed `generateStream` attempt on the ~1049-node template. **So today the
  honest recommendation is: decode runs best LOWERED** (or with build-from-IR skipped
  for a known-uncovered decode template); the compiled-forward win is **gated on the
  RoPE generator**, not on arg-reduce.
- **The design's P2 projection ("the full win is here, compiled") is NOT yet realized**
  — it is one bounded generator (RoPE-offset-as-data) away. What IS realized and
  measured: the host-tax amortization (~1.8× on A100, a strict lower bound for the
  browser, where the fence is a larger share of the per-token wall).

### Gates run

`tools/gate-wall.sh --profile training` (green); `test/argmax-strided-view.spec.ts`
(argmax op, green); `tools/t-uk-generators-parity.ts` (the P2 generator compiled-parity
gate, PASS — argmax + max/min + RMSNorm compiled == lowered, cutover engaged);
`tools/t-uk-block-diff.ts` (mother gate + the new **compiled arm**: build-from-IR
ENABLED == DISABLED == host across the P1 matrix — partial coverage is faithful, all
PASS); `tools/t-uk-feedback.ts` re-run green; decode paths byte-unchanged when
`TORCHLETTE_UNROLLED_K` is off (block branch compile-time-guarded); strict-lifetime
default. **Weight-norm delta:** `src` SLOC +~150 (the three generators +
`planArgReduceDispatch`/`planRMSNormForwardDispatch` + the max/min routing), +1 debug
flag (`TORCHLETTE_DEBUG_CENSUS`, the census diagnostic — a `DEBUG_*` category flag,
mirroring `DEBUG_COMPILED`).

### What P3 (Gumbel sampling) needs from RNG-as-data — unchanged, restated

P3's on-device categorical is `argmax(logits/temp + gumbel)`, `gumbel = −log(−log(U))`,
`U = randF32(seed_as_data, offset)` (§3.5). The arg-reduce generator this pass lands is
**exactly the terminal op** of that expression, so P3's selection already compiles; its
net-new need is the **seed/counter as a graph-threaded DATA uniform** so a replayed
block advances RNG deterministically (the same scalars-as-data pattern as Adam's
`step_size`). Notably, that need is the **same shape** as the RoPE-offset-as-data
blocker above — both are per-replay-varying uniforms that must flow as data, not baked
into the recording. Closing the RoPE volatile-config repack builds the exact mechanism
P3's seed-as-data reuses.

---

## P3' STATUS — the RoPE generator (block compiles → the REAL multiplier), Gumbel sampling, the failed-gen memo (2026-07-20)

*Worktree off `main@0b4d4900` (P2 HEAD). Staged commits, no push. dw-2-1 (A100),
random-init Qwen3 (the static-KV `forwardStatic` path), node v22 + Dawn/Vulkan
vk-shim, device reserved via `tools/pick-gpu.sh`. Reproduce: `eval "$(tools/pick-gpu.sh)";
LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-multiplier.ts` (the
multiplier), `… tools/t-uk-gumbel-parity.ts` (Gumbel), `… tools/t-uk-block-diff.ts`
(mother gate + compiled arm), `TORCHLETTE_DEBUG_CENSUS=1 … t-uk-block-diff.ts` (census).*

### The census correction — the P2 `{fusedRoPE}` attribution was right on the OP, wrong on the WHY

The P2 census named `{fusedRoPE}` as the lone residual and attributed it to "RoPE's
per-position offset is a per-replay-varying uniform" (the frozen-uniform class). The
**runtime census on the actual `forwardStatic` path falsifies the WHY**: on the
static-KV decode path the RoPE cos/sin are **`gather` results**, not narrow views —
the model deliberately routes the position through the gather's INDEX TENSOR as DATA
(`model.ts` `buildRope`: "position enters as an INDEX TENSOR … so the plan template
stays stable across decode steps") precisely BECAUSE a narrow view's per-token `start`
is payload the template fingerprint hashes. A gather output is contiguous+offset-0, so
**the RoPE config's `cos_offset`/`sin_offset` are 0 on this path** — the position never
enters the RoPE config at all; it is already covered as data in the gather index.
fusedRoPE was uncovered simply because it had **no generator** in `generateSequential`,
not because of a varying offset. (Prior-agent attributions are hypotheses, not facts —
`docs/agent-ops.md`. The runtime census, `TORCHLETTE_DEBUG_CENSUS`, was the referee.)

### The RoPE mechanism as landed — offset-as-volatile-data, unconditional (the seam)

`generateRoPE` (`src/executor/stream-generate.ts`) + `planRoPEDispatch` /
`ropeVolatilePack` (`src/backend/webgpu/rope-kernel.ts`, single-sourced with the
imperative `dispatchRoPE` through the same `ropeTileKernel`) emit, per node:
`ALLOC(output) + a per-node params slot (the tile config as a UNIFORM buffer) +
TAG_UNIFORM(volatile offset repack) + DISPATCH [input, cos_table, sin_table, output,
config]`.

- **How the offset flows.** The tile config's `{seq_len, head_dim, sin_scale,
  cos_offset, sin_offset}` are re-derived from the CURRENT node every replay by a
  volatile `TAG_UNIFORM` (`ropeVolatilePack` → the instance's `volatilePack` → the same
  `packUniforms` the imperative dispatch writes; the offsets from `directInputOffset`,
  single-sourced with #71's view-meta deriver). This is the compiled-plan-volatile-
  uniforms precedent (`setAdamConfigUniforms` the exemplar). A per-node params slot (not
  the shared keyed config buffer) is used deliberately: distinct buffers per rope node
  avoid the `volatileRewriteNeedsFlush` storm a shared config would trigger across the 4
  rope dispatches/step.
- **The seam assertion — no positional value bakes.** The volatile repack is emitted
  **UNCONDITIONALLY**, so the record-time offset is overwritten on every replay by
  construction — a positional value can never freeze. On the gather path the repack
  writes 0 (position is in the gather index); on any contiguous-narrow path it re-derives
  the real offset. qk (input 0) is raw-bound flat-from-0 (`CONTIGUOUS_OPERANDS ["fusedRoPE",
  [0]]`); cos/sin need only contiguous STRIDES (offset folded), asserted by
  `refIsContiguousStrides` — which handles the RELEASED-storage case (decode cos/sin are
  gather intermediates freed before plan-build) by deriving contiguity structurally (a
  non-view producer allocates contiguous+offset-0 by construction); a released view
  producer bails to lowered rather than fold into a possibly-strided table.

### The census after RoPE — EMPTY (steady-state fully covered) → the block COMPILES

Runtime census on the K=8 block (`~1049`-node plan, one force): **steady-state uncovered
= `{}`**. The only census lines are the FIRST-EXECUTION artifacts
(`op:*[config-missing]`, `fused[no-input-pattern]`) that `populateCapturesFromIR`
captures on the 1st run and covers on the 2nd+ — never logged at steady state. The block
reaches full coverage → **compiled forward replay** (`getCompiledStreams()>0` on every
`block-comp` multiplier cell). The mother gate (`t-uk-block-diff`) is byte-identical
compiled==lowered==host across the full P1 matrix.

### THE REAL MULTIPLIER (measured, honest — random-init Qwen3 8L/512d, N=64, 3 repeats, medians)

| K  | block-low ms/tok | subs | block-def ms/tok | subs | host/low | host/def | low/def |
|----|-----------------|------|-----------------|------|----------|----------|---------|
|  4 |           61.96 |  144 |          **24.26** |  144 |    0.56× |  **1.43×** | **2.55×** |
|  8 |           56.49 |  120 |          **30.81** |  120 |    0.61× |  **1.12×** | **1.83×** |
| 16 |           59.60 |  112 |           68.88 |  112 |    0.58× |    0.50× |   0.87× |

*host per-token loop: 34.62 ms/tok, 192 submits (K-independent). `getCompiledStreams>0`
on every block-comp cell = YES.*

- **block-def** = build-from-IR ENABLED (the shipping DEFAULT). With the RoPE generator
  the block fullyCovers and reaches **compiled forward replay** — **`low/def = 2.55×` at
  K=4** (compiled 2.55× faster than the lowered block) and **`host/def = 1.43×`** (faster
  than the per-token host loop). **This is the compiled-forward win the P2 design projected,
  now REALIZED** — RoPE coverage was the single gate, exactly as P2 predicted (the WHY was
  "add a generator", not "offset-as-data", but the offset-as-data is the correct robust
  mechanism and it is what landed).
- **P2's honest verdict is INVERTED.** P2 measured `low/def ≈ 0.77×` (build-from-IR
  net-NEGATIVE — every block wasted a failed generateStream attempt on the uncovered
  ~1000-node template). Post-RoPE that tax is gone and the compiled replay is the win.
- **block-low** = build-from-IR DISABLED (`TORCHLETTE_COMPILED_PLAN=0`): the pure-lowered
  floor. It is slower than host here only because host decodes WITH build-from-IR (its
  per-token forward compiles) — block-low is a lowered-vs-compiled control, not a shipping
  path.
- **The K=16 regime regresses** (block-def 68.9 ms/tok, low/def 0.87×) — reproducible
  across both runs. The larger unrolled template (~2097 nodes, 16 steps of activations
  live at once) raises the compiled per-token overhead above the K=4/8 sweet spot; the
  cause (memory-plan pressure vs a partial-recompile) is un-root-caused. **The sweet spot
  is K∈{4,8}.** Named, not hidden.
- **Browser projection:** the per-token host tax (fence + JS dispatch + slower readback
  bus) is a larger share of the per-token wall in-browser, so `host/def` is a strict LOWER
  bound for the browser amortization win.

### Gumbel-max sampling (P3) — on-device, deterministic, seed-as-DATA

`decodeBlock`'s `opts.sample = { temperature>0, seed }` makes the feedback selection
on-device **Gumbel-max**: `id = argmax(logits/temperature + g)`, `g = -log(-log(u))`.
The uniform `u` is the seed-as-DATA channel (§3.5 explicitly sanctions "an uploaded
uniform/tensor"): `gumbelUniform(seed + absolutePosition, V)` (a mulberry32 host PRNG,
the SINGLE SOURCE both `decodeBlock` and the parity reference draw from) is uploaded as a
tensor, and the transform + `argmax` close on-device (no per-token readback inside the
block). The seed is **position-canonical** (`seed + absolutePosition`), so it is
byte-reproducible across replay AND aligns a per-token host loop's seeds with the block's
regardless of K/bucketing. `temperature===0` stays the greedy `argmax(logits)` path,
unchanged. Full-vocab top-p / arbitrary host samplers remain the typed K=1 host residue
(§4).

*Why uploaded-uniform, not on-device `randF32`:* the on-device RNG data-source
(`api.rand`) tripped a lowered-path buffer-destroy in the block's forced graph, and the
§3.5 "uploaded uniform/tensor" pattern is both the sanctioned channel AND cleaner
(host-controlled determinism, no RNG-buffer lifetime). The GPU-Philox variant (seed as a
tiny uniform, noise generated on-device) remains a possible future optimization gated on
`rand`-as-data coverage — the SAME shape as any per-replay-varying uniform — but is not
needed for correct, deterministic sampling.

**Gate `tools/t-uk-gumbel-parity.ts` (all PASS):** (1) PARITY — the sampled block is
byte-identical to a per-token host-loop reference drawing the same per-position seeds and
the same on-device Gumbel-max, over K∈{1,4,8} × short + bucket-crossing prompts; (2)
DETERMINISM — same seed decodes byte-identically twice; (3) SEED SENSITIVITY (T=30, gumbel
-dominated so the argmax-dominated tiny model actually varies) — different seed ⇒ different
stream; (4) TEMPERATURE 0 == greedy; (5) GUMBEL FORMULA unit — device
`argmax(logits/temp + -log(-log(u)))` == host, wide-margin.

### The failed-generation-tax — memoed for the GENERAL class (sized, not forced)

The RoPE coverage removes the tax for the decode block (it now fullyCovers). For the
GENERAL class — any genuinely-uncovered RECURRING template (a strided view, a >128MB
chunked path, a non-f32 dtype) that re-enters the build block every step and re-pays the
whole ~N-node `generateStream` walk for a verdict that never changes — a cheap memo lands
(`executor.ts` `buildFailures` / `BUILD_FAILURE_MEMO=4`, no flag): after 4 consecutive
coverage misses a template's structure will not become coverable mid-process (generators
are static; templateFp hashes structure), so the verdict is memoed and the build block
skipped straight to lowered. The threshold sits ABOVE the warmup horizon (config-missing /
no-input-pattern artifacts resolve by the 2nd execution, and a fullyCovered result deletes
the counter), so a truly-coverable template is never stranded. Always-on, strictly a saving.

### The one blemish — RESOLVED (the compiled-plan external-buffer-destroy transient; P4 precondition met, 2026-07-20)

Making the block compile newly exercised a **bounded, non-corrupting** compiled-plan
external-lifetime transient: a compiled plan bound an EXTERNAL materialized input
(`kind=external ref=materialized`, the block's/host-loop's per-step host-token `idx`) that
had been destroyed → "used in submit while destroyed", the dropped-submit class (CLAUDE.md).

**Root cause (single-variable, deterministic repro `TORCHLETTE_DEBUG_DESTROYED=1
TORCHLETTE_STRICT_GPU=1 … t-uk-block-diff.ts`).** The `idx` upload's result is HARVESTED
into a co-owned planner-registry entry buffer (cross-plan temp packing), and that same
physical buffer is what the NEXT forward plan resolves as its slot-0 external input
(`getInputStorage` → `gpuBuffer(storage.backendTensor)`). When the producer template is
invalidated at a step boundary (`observeStepBoundary` → the compiled-invalidator guard →
`destroyCompiledPlanBuffers`), the registry entry's last plan-owner drops, so the entry
buffer is freed — **while the live `idx` storage still backs it and the consumer plan is
about to bind it.** The existing park-live guard (`liveHarvestIdsForBuffer`) that should
have caught this had an `isRegisteredStorage` gate: it parked ONLY `registerState`'d
post-boundary readers (the whole-step deferred loss), excluding the plain materialized
harvest storage the decode block relies on. Proof: at the destroy, the plan's
`_lastHarvestIds` contained the idx storage `1040:dead=false:reg=false:buf=531` — ALIVE,
backing the freed buffer `#531`, but skipped by the registered gate.

**Fix (smallest honest, at the seam — `compiled-plan.ts` `liveHarvestIdsForBuffer`).** Drop
the `isRegisteredStorage` gate: a registry-entry buffer is PARKED (kept pinned, reclaimed
once its readers die via `reclaimParkedLiveBuffers`) while ANY live storage backs it
(identity match on the current backing buffer), not only registerState'd ones. Genuinely
-dead entries still free promptly (empty live set → destroy, unchanged). The block-diff
gate now asserts `getGpuUncapturedErrorCount() === 0` (failing-first: 2 → 0).

**Verified green:** `t-uk-block-diff.ts` under `DEBUG_DESTROYED + STRICT_GPU` — **zero
destroyed-binds, zero uncaptured GPU errors, VERDICT PASS** (was 2 dropped submits);
`gate-wall --profile training` **PASS=15 ENV=0 REAL=0** (test:gates, parity-fullstack, tape
matrix, 124M-regression, ledger, checkpoint-seg — strict-lifetime default throughout);
multiplier spot-check UNCHANGED (K=4 low/def 2.91×, host/def 1.46×). **P4 (demo cutover /
default-on) is UNBLOCKED.**

### Gates run

`tools/gate-wall.sh --profile training`; `test/argmax-strided-view.spec.ts`;
`tools/t-uk-block-diff.ts` (mother gate + compiled arm, byte-identical); the census
(`TORCHLETTE_DEBUG_CENSUS=1` → steady `{}`); `tools/t-uk-generators-parity.ts`;
`tools/t-uk-gumbel-parity.ts` (all PASS); `tools/t-uk-multiplier.ts` (the table above);
`tools/t-uk-feedback.ts`; decode paths byte-unchanged when `TORCHLETTE_UNROLLED_K` is off
(block branch compile-time-guarded); strict-lifetime default.

---

## P4 STATUS — THE DEMO CUTOVER: greedy default-on, steering composes, sampled opt-in (2026-07-20)

*Worktree off `main@48cda47b` (P3' HEAD). Staged commits, no push. dw-2-1 (A100),
random-init Qwen3 (packages/qwen3-browser, the static-KV `forwardStatic` path),
node v22 + Dawn/Vulkan vk-shim, device via `tools/pick-gpu.sh`. Reproduce:
`eval "$(tools/pick-gpu.sh)"; LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH
npx tsx tools/t-uk-cutover.ts` (the cutover routing gate), `… tools/t-uk-steering-diff.ts`
(the steering differential), `… tools/t-uk-block-diff.ts` (mother gate),
`… tools/t-uk-gumbel-parity.ts` (Gumbel logic).*

### The cutover as landed — GREEDY default-on (opt-OUT), the flag inverted

`generateChat` (both `packages/qwen3-browser` AND `packages/gemma2-browser` — the
gemma2 block was the P1-deferred "mechanical port", now landed) routes **GREEDY
decode through the unrolled-K block BY DEFAULT**. `unrolledKFromEnv()` inverted to
opt-OUT: **UNSET → the block, `K = UNROLLED_K_DEFAULT = 4`**; **`0`/`1` → the
per-token host loop** (the pre-cutover decode, the soak escape hatch); **`≥2` →
the block at that K**.

- **K DEFAULT = 4 (justified from the P3' multiplier).** K=4 is the sweet-spot
  cell with the best compiled win (`host/def 1.43×`, `low/def 2.55×`) AND the
  finest streaming granularity (K tokens per readback); K=8 is the other sweet
  cell (`host/def 1.12×`); **K=16 REGRESSES** (`host/def 0.50×`, un-root-caused
  memory-plan pressure). Default clamped to the low end of {4,8}.
- **The residue (per-token host loop), by §4:** any **top-k / top-p / nucleus**
  sampler (a full-distribution host read the block cannot express), **default
  (unflagged) temperature sampling**, and any **non-static-KV** path. **The
  shipped demo samplers are exactly this residue** — both `examples/qwen3-steering`
  and `examples/gemma2-sae-demo` decode with `temperature 0.7, topK 20/40,
  topP 0.95`, so their runtime decode stays on the host loop (byte-unchanged
  pre/post cutover). The cutover changes the DEFAULT for the samplers the block
  covers on-device; it does not silently alter the demos' sampling distribution.
- **Sunset:** `TORCHLETTE_UNROLLED_K` born 2026-07-19 (opt-in), **default-on
  2026-07-20 (this pass)**. No NEW flag; the opt-out reuses the existing one.
  P6 removes the K=1 host loop for the covered samplers and retires the knob.

### The steering hook composes with the block — BYTE-IDENTICAL (the load-bearing proof)

The live consumers' core behavior is a residual-stream steering hook threaded into
`model.forward` every step (`steering.ts makeResidualHook`: `x += α·dir` at layer
L). Post-cutover that hook applies per-step **inside the block's graph**.
**`tools/t-uk-steering-diff.ts` (VERDICT PASS)** proves the composition exact,
using the demo's actual additive-steering mechanism:
- **GREEDY composition**: hooked block ids **byte-identical** to the hooked
  per-token host loop over 2 prompts × K∈{1,4,8,16} × a bucket crossing (all PASS).
- **Steering is LIVE (α≠0)**: the hooked stream **differs** from the unsteered
  (α=0) stream — so the composition proof is real, not a no-op agreeing with a
  no-op. Unsteered block == unsteered host (baseline intact).
- **α scales**: α=2 stream ≠ α=16 stream, and each still == its host loop —
  the steering magnitude flows through the block unchanged.
- **COMPILED vs LOWERED with the hook active**: hooked block build-from-IR
  ENABLED (default) == LOWERED == host — steering composes with compiled replay.
- **Zero uncaptured GPU errors** across the greedy composition.

### The sampled (Gumbel) cutover — opt-IN, gated on a PRE-EXISTING transient (named blocker)

The Gumbel sampled block (§3.5, on-device `argmax(logits/temp + gumbel)`) is
**wired end-to-end** in `generateChat` (routing + a Gumbel-consistent prefill
token, `gumbelPrefillToken`) but is **opt-IN via an explicit flag
(`unrolledKExplicit`), NOT default-on**. Reason — a **P4-discovered PRE-EXISTING
dropped-submit transient** in the sampled block path:

- **Symptom:** `t-uk-gumbel-parity.ts` (which passes its ids-match verdict) emits
  a `[Buffer] used in submit while destroyed` uncaptured GPU error — it never
  asserted `getGpuUncapturedErrorCount()===0`, so the transient shipped unnoticed
  at P3'. `t-uk-steering-diff.ts`'s sampled arm surfaces it as **actual
  corruption** (the dropped submit's stale buffer feeds a load-bearing read →
  sampled ids MISMATCH). It is timing-dependent (gumbel-parity's ids happened to
  be unaffected; the steering ordering's were not) — the classic silently-wrong
  class CLAUDE.md warns of.
- **Root cause (traced, `TORCHLETTE_DEBUG_DESTROYED=1`):** the destroyed bind is
  `kind=external ref=materialized storage=945 buf=b1` — the block's **first-token
  upload** (`api.tensorFromArray([lastTok],[1,1])`), a materialized EXTERNAL, not
  a harvest RESULT. The P3' park fix (`liveHarvestIdsForBuffer` /
  `compiled._lastHarvestIds`) parks a torn-down registry-entry buffer only when a
  live **harvest RESULT** backs it; it does not track external-INPUT uploads. In
  GREEDY the block's `idx` feed IS the reshaped selection RESULT (a harvested id),
  so it is parked — which is why greedy is STRICT_GPU-clean (`t-uk-block-diff`
  zero errors). The sampled path's extra per-step `u` uniform upload shifts buffer
  reuse so the un-parked first-token upload's buffer is destroyed while a consumer
  plan binds it.
- **Ruling:** fixing this is a real extension to the harvest-park mechanism in
  the most delicate part of the runtime (the fence-gated destroy / cross-plan
  buffer-lifetime class that has bitten this project repeatedly); per
  `docs/agent-ops.md` (reproduce-before-theorizing; no blind lifetime fixes) it
  is a **named, bounded follow-on**, not a same-pass patch. Shipping the sampled
  block default-on would ship the corruption, so it is opt-in until the park
  mechanism tracks external-input uploads (the fix: fold the phase-1 external
  materialized storages into the park set, or park on the upload buffer's
  registry-entry identity). The Gumbel SELECTION LOGIC is proven correct
  (gumbel-parity's block==host); only its lifetime is gated.

### Perf on the record (this A100 box, random-init Qwen3 8L/64d)

The mother gate's block-vs-loop economics (N=16, K=8, lowered):

| arm | submits | ms/tok |
|---|---|---|
| host per-token loop | 48 | ~110 |
| **unrolled block (greedy)** | **15** | **~26** |

**~3.2× fewer submits, ~4.2× faster ms/tok** on this tiny config — the host-tax
amortization measured honestly. The compiled multiplier at demo scale is the P3'
table (`host/def 1.43×` @ K=4, `getCompiledStreams>0`), still the authoritative
number; the browser projection is a strict LOWER bound on `host/def` (the per-token
host tax — fence + JS dispatch + slower readback bus — is a larger share of the
per-token wall in-browser). **How to measure the browser number** (Vin's to run
when next driving a demo): the demos read `TORCHLETTE_UNROLLED_K` via
`globalThis.__TORCHLETTE_ENV__` (browser hook) — set it to `0` (host) vs unset
(block K=4) across two runs and compare `stats.tokPerSec` / `decodeBreakdown`; a
`?unrolledK=<K>` URL param wiring the same global is the one-line UI hook.

### Gates run (all green unless noted)

`tools/t-uk-cutover.ts` (P4 routing gate, PASS — greedy default→block
byte-identical to opt-out host; explicit K→block; top-k+top-p→host residue; zero
GPU errors); `tools/t-uk-steering-diff.ts` (PASS — greedy hooked composition +
α-live + α-scale + compiled==lowered + zero GPU errors; sampled arm is a logged
characterization of the transient); `tools/t-uk-block-diff.ts` **under
`TORCHLETTE_STRICT_GPU=1`** (mother gate + compiled arm, byte-identical, zero
errors — did not crash); `tools/t-uk-gumbel-parity.ts` (Gumbel logic block==host,
PASS verdict; the sampled transient noted above); `tools/gate-wall.sh --profile
training` (test:gates, parity-fullstack, tape-matrix ON≡OFF, ledger,
checkpoint-seg — strict-lifetime default throughout); decode byte-unchanged when
`TORCHLETTE_UNROLLED_K=0` (opt-out) or for the top-k/top-p residue. **Weight-norm
delta:** `src` SLOC **unchanged** — the cutover lives entirely in `packages/`
(admission pressure; packages/ is not the src diamond) and adds NO new env flag
(opt-out reuses `TORCHLETTE_UNROLLED_K`). New tools: `t-uk-cutover.ts`,
`t-uk-steering-diff.ts`.

### P4 acceptance vs the phase plan (§5)

§5 P4's gate — "the demos' token streams byte-identical pre/post cutover; browser
suite green; SAE-steering interactivity unregressed" — is met **for the greedy
covered path** (byte-identical, steering composes) and **trivially for the demos'
top-k+top-p sampler** (it stays the host residue, path byte-unchanged). §5 P4 also
says "remove the K=1-per-token host loop for the COVERED samplers" — that removal
is P6's, and it is bounded to GREEDY here: **the demos' sampler is the §4 residue,
so their host loop is NOT removed by this cutover.** This is the honest state of
"the cutover frees the live consumers": the greedy path is freed; the demos' host
loop persists as long as they sample with top-k/top-p — which forks P5 (below).

---

## P5 — THE LEDGER EXECUTION (declared next, with its absence-proof checklist)

The P4b ledger (`step-function-compiler-design.md` §5: obs-liveness 807 + step-tape
820 + replay 680 + step-object 156 + cross-plan-edges 152 + tape-profile 18 =
**−2633** outright, + **−1100…−1900** partial) executes ONLY once decode is
proven **no longer a consumer** of the tape / observed-liveness / cross-plan-edges
— per-item, each with a **consumer-ABSENCE re-proof** mirroring the P4b
presence-proofs (Objection 3 / risk 1: no file deleted while a decode demo routes
through it). P5's entry checklist:

1. **The consumer fork must be closed FIRST.** P4 freed GREEDY decode to the block,
   but the **shipped demos still route their top-k+top-p sampling through the
   per-token host loop** (the CapturedFn + tape). So the tape's decode consumer is
   NOT yet gone. P5 is BLOCKED until one of (forks §8 Q1, Vin's call):
   - (a) the demos adopt a block-covered sampler (greedy or on-device Gumbel), OR
   - (b) on-device small-top-k + top-p lands (§4 net-new: workgroup `inclusiveScan`
     prefilter + Gumbel; §8 Q1), moving the demo sampler onto the block, OR
   - (c) the sampled-block **transient is fixed** AND the demos switch to Gumbel.
   Until then the P4b STOP rule holds: a proven decode consumer remains.
2. **Absence re-proofs (run under unrolled-K, greedy-and-block-sampler decode):**
   - **`tools/t-p4b-decode-edges.ts` under unrolled-K** ⇒ expect
     `crossPlanEdgeStats().producers = 0` and no observed-liveness convergence on
     the decode path (the P4b PRESENCE proof was `producers=1, convergedTemplates=1`;
     the deletion gate is BOTH → 0 because the block STATES its liveness).
   - **`tools/t-decode-whole-step.ts`** ⇒ expect **Verdict A** (decode traces +
     compiles as a distinct static function), inverting P4a's Verdict B.
   - **Per-item consumer-absence** for each file before deleting it, decode AND
     training: `cross-plan-edges.ts` (152) → `tape-profile.ts` (18) →
     `step-object.ts` (156) → `step-tape-replay.ts` (680) → `step-tape.ts` (820)
     → `observed-liveness.ts` (807), plus the partial harvest/arena/pool
     reductions — each behind a green parity gate, the retained two-plan-eager
     path + `CHECKPOINT_EAGER_REFUSAL` (P4b PERMANENT POLICY, a TRAINING concern)
     untouched.
3. **The fix P5 also wants from P4's residue:** the sampled-block park-external
   fix (above) is the same lifetime discipline the ledger's harvest/arena
   partial reductions touch — closing it feeds P5's harvest-seam re-proof.

The covenant reaches net-negative when this ledger executes (§5 projected
`src` **−3100…−3600 vs pre-Everest**); P4 is the cutover that unblocks the greedy
half of the consumer removal, and names precisely what still gates the rest.

---

## Appendix — reproduction

- Probes (this worktree, `main@1cc28d3e`): `tools/t-uk-feedback.ts` (P1),
  `tools/t-uk-scale.ts` (P2), `tools/t-uk-economics.ts` (P3), `tools/t-uk-eos.ts` (P4).
- Run: `VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim npx tsx tools/t-uk-<probe>.ts`.
  Pretrained arm (P1): `UK_PRETRAINED=1 UK_MODEL_DIR=<abs>/models/distilgpt2`.
- Numbers: dw-2-1 (A100), device 10, distilgpt2, node v22 + Dawn/Vulkan vk-shim.
- Parent: `docs/step-function-compiler-design.md` (P4a Verdict B, P4b covenant STOP,
  §5 deletion ledger). The static-KV substrate: `packages/qwen3-browser/src/model.ts`
  (`forwardStatic`, `StaticKV`, `KV_BUCKET`).
