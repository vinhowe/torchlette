# Checkpoint E — intrinsic-ladder game loop

All targets are `ceil(known solved napkin cost × 1.10)`. H/M are local-step
element counts, consistent with the existing surface and F2.

| Exercise | Baseline | Known solved cost | Target | Intended moves |
| --- | --- | --- | --- | --- |
| 1 · Fuse the chain | H₁ 8,192 | H₁ 4,096 | H₁ ≤ 4,506 | paint A, paint B |
| 3 · Carry the moments | H₁ 6,144; M₁ 2,048 | H₁ 256; M₁ 256 | H₁/M₁ ≤ 282 | stream jam, Welford, paint A/B, stream r=128 |
| 8 · Cross the lemma wall | H₁ 12,288; M₁ 4,096 | H₁ 256; M₁ 256 | H₁/M₁ ≤ 282 | stream jam, online softmax, paint A/B, stream r=128 |
| 9 · Assemble FlashAttention | H₁ 14,155,776; M₁ 6,291,456 | H₁ 147,456; M₁ 98,304 | H₁ ≤ 162,202; M₁ ≤ 108,135 | stream jam, lemma, fuse S/P, group q=64, stream x=32 |

## Captures

- Level 1: [goal](./checkpoint-e-level1-goal.png),
  [paint-only vocabulary](./checkpoint-e-level1-jam-na-vocabulary.png),
  [first fusion](./checkpoint-e-level1-lemma-na-first-fusion.png),
  [completion](./checkpoint-e-level1-completion.png). Jam and lemma are
  deliberately not applicable at this rung; the filenames preserve the common
  four-checkpoint review matrix.
- Level 3: [goal](./checkpoint-e-level3-goal.png),
  [dependency jam](./checkpoint-e-level3-jam.png),
  [Welford inspection](./checkpoint-e-level3-lemma-inspection.png),
  [completion](./checkpoint-e-level3-completion.png).
- Level 8: [goal](./checkpoint-e-level8-goal.png),
  [lemma wall](./checkpoint-e-level8-jam.png),
  [m/ℓ and correction inspection](./checkpoint-e-level8-lemma-inspection.png),
  [completion](./checkpoint-e-level8-completion.png).
- Level 9: [goal](./checkpoint-e-level9-goal.png),
  [capstone wall](./checkpoint-e-level9-jam.png),
  [reused lemma inspection](./checkpoint-e-level9-lemma-inspection.png),
  [FlashAttention completion](./checkpoint-e-level9-completion.png).

`pnpm test` runs all unit tests, the prior 12-assertion canvas protocol, and a
four-level Playwright path. The game test asserts pre-jam lemma absence,
post-jam presence, level-1 vocabulary gating, inspection contents, target
completion, and completion-ledger recording.
