# Cleanliness audit — 2026-07-23 (seven independent auditors, cross-checked)

Seven independent Opus auditors, each calibrated to the platonic bar (no credit
for effort; measured-and-fenced compromises free; undocumented ones double), one
denied all documentation. Every load-bearing claim was cross-checked by the
orchestrator; one major finding was falsified (noted below).

## Scores
| Area | Score | One-line verdict |
|---|---|---|
| Frontend/API | 7/10 | Table-driven core is real; phantom union types, 15-method lifetime surface, api-everywhere tax |
| Semantic stratum | 8/10 | Derivation claim TRUE at core (traced); composite backwards + attention are the honest hand frontier |
| Engine/IR | 8/10 | Frozen-scalar disease structurally cured; VIEW_OPS drifted twice; name-lists instead of op properties |
| Executor | 8.5/10 | No wrong-not-slow fallback exists (verified); 165 typed refusals; vestige = dead tape branch, 4k-line file |
| Backend/lifetime | 8/10 | One layered model, not five patches; 2 sequencing invariants are PROSE+ordering only |
| Fresh-eyes legibility | 5/10 | Spine reconstructible cold in ~1h; key contracts live in changelog prose; no public custom-op seam |
| Coherence/meta | 7.5/10 | Deletions were honest; drift concentrated in 3 false DESIGN headers + 3 ledger rows + stale CLAUDE.md paths |

**Composite: ~7.5/10 against the platonic bar.** The architecture's load-bearing
claims survive adversarial verification: derivation is real, the compiled path
cannot be silently wrong, memory is one model, refusals are typed. The distance
to ideal is concentrated in (1) the underived remainder, (2) two prose
invariants, (3) legibility-without-docs, (4) self-description drift.

## Cross-check corrections (auditor claims falsified/adjusted)
- **FALSE**: backend auditor's "deviceTopK is dead, zero callers, delete" — live
  consumers in packages/qwen3-browser, packages/gemma2-browser, three t-uk-topk
  tools (auditor searched src/+test only). Do NOT delete.
- CONFIRMED by orchestrator: Triton GELU constant divergence (triton-emit.ts:181
  `0.7978845608028654` vs erf.ts:36 `0.7978845608`); step-tape suppression flag
  set only by test/observed-liveness.spec.ts; stream-generate VIEW_OPS local set
  (with 4 non-opcode phantom entries); CLAUDE.md references nonexistent
  src/engine/ and deleted custom-backward.ts (3×).

## The consolidated fix ladder (by leverage-per-line)
1. **assertQuiesced()** at releaseStepTemps/destroyUnreachable entry (~5 lines):
   promotes both prose sequencing invariants to deterministic CPU throws. The
   highest remaining tail risk in the repo's most catastrophic class.
2. **Single-source the view predicate** (~20 lines): one exported VIEW_OPS
   derived from the canonical opcode list; delete the 4 phantom entries; closes
   the only merely-believed load-bearing engine invariant.
3. **Excise the dead step-tape suppression** (setStepTapeReplayActive + branches
   + ~10 stale comments): the largest vestige pocket, sitting on the strict-
   lifetime guards.
4. **Doc-reality CI check** (mirror of the weight-norm hook): ledger-row ↔ src
   flag bijection; doc-referenced paths resolve; vitest includes resolve.
   Catches every drift class the coherence audit found, mechanically, forever.
   Plus the one-time fixes: 3 false "Status: DESIGN" headers, DERIVED_ADAM/
   PACKED_OPT/PACK_MAX_BYTES ledger rows, CLAUDE.md paths, 2 dead vitest
   includes, erf.ts stale comment, import the Triton constant.
5. **Legibility package** (the fresh-eyes prescription): a code-level glossary
   (plan/template/harvest/demotion/island/witness/skeleton/realizer); ONE named
   demotion predicate; ONE replayPathFor() decision function; convert the two
   worst changelog-comment clusters into stated invariants.
6. **Engine-optional API** (the frontend #1): api as optional last arg
   defaulting to the torch singleton — the pattern the optimizers already prove.
   Plus: type all reductions Tensor; throw on backward-of-non-grad; demote the
   lifetime machinery methods to underscore; GradScaler structural Optimizer type.
7. **The two missing folds** (the semantic frontier, a real campaign):
   (a) Expr→BlockExpr fold derives the forward GPU activations (deletes the
   twice-written DSL bodies); (b) CompNode-adjoint pass derives the layernorm/
   rmsnorm/softmax/CE backwards — the largest remaining hand arithmetic, checked
   today only by oracle, not by construction. Attention stays hand (fused-kernel
   -first, oracle-fenced) but its exclusion should be STATED in the stratum.
8. Smaller named items: NON_CSE as op property not name list; rename BaseState
   (dead ssa/loc union); executor.ts + stream-generate.ts splits; named
   RECLAIM_INTERVAL_NODES with provenance; import the 128MB constant at 25
   sites; conv2d delete-or-sunset; layering fix for backend↔optim cycle;
   tailwind deps out of root; public custom-op-with-gradient seam (mission-
   relevant: the editor story wants it).

## What the audits proved is already platonic (keep-verbatim consensus)
contraction.ts; the two OptTerm folds + typed mm refusal; index-map's transpose
fact; OP_REGISTRY-derived dispatch; computePlanFingerprint + PAYLOAD_HASH_EXEMPT
+ thrash detector; LazyIRNode result/results brand; refEquals; the miss()/
fullyCovered coverage machinery; assertNoGeneratorLeaf; the memory planner's
audit block; getInputStorage's throwing seam; _derived owner-set classification;
canRecycle + idempotent release; the single WGSL chokepoint; the reduction
monoid-as-data with the left-biased where-fold.
