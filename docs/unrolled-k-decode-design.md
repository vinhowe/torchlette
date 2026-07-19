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

- **P2 — Compile + replay the block; measure the real multiplier.** Route `S_dec`
  through the whole-step build-from-IR compiler + global memory planner (it is a static
  single-force graph — the compiler's native diet). Gate: the block reaches compiled
  replay within a bucket (template converges, unlike growing-KV Probe 2), memory flat,
  and the compiled ms/tok beats the 30.8 ms lowered floor. Re-measure the economics
  (Probe 3/4 arms) compiled + **in-browser** (the real target). *First honest speed
  win beyond the 1.6× host-tax amortization.*

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

## Appendix — reproduction

- Probes (this worktree, `main@1cc28d3e`): `tools/t-uk-feedback.ts` (P1),
  `tools/t-uk-scale.ts` (P2), `tools/t-uk-economics.ts` (P3), `tools/t-uk-eos.ts` (P4).
- Run: `VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim npx tsx tools/t-uk-<probe>.ts`.
  Pretrained arm (P1): `UK_PRETRAINED=1 UK_MODEL_DIR=<abs>/models/distilgpt2`.
- Numbers: dw-2-1 (A100), device 10, distilgpt2, node v22 + Dawn/Vulkan vk-shim.
- Parent: `docs/step-function-compiler-design.md` (P4a Verdict B, P4b covenant STOP,
  §5 deletion ledger). The static-KV substrate: `packages/qwen3-browser/src/model.ts`
  (`forwardStatic`, `StaticKV`, `KV_BUCKET`).
