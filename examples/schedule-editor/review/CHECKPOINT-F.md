# Checkpoint F — game rebuild after self-playtest

Viewport: 1600×1100, Chromium, revised Commit C candidate.

Each level has four review states:

| Lesson | Goal | First action | Aha | Completion |
|---|---|---|---|---|
| Level 0 · The long way home | `checkpoint-f-level0-goal.png` | `checkpoint-f-level0-first-action.png` | `checkpoint-f-level0-aha.png` | `checkpoint-f-level0-completion.png` |
| Level 1 · Stop the round trips | `checkpoint-f-level1-goal.png` | `checkpoint-f-level1-first-action.png` | `checkpoint-f-level1-aha.png` | `checkpoint-f-level1-completion.png` |
| Level 3 · A useful backpack | `checkpoint-f-level3-goal.png` | `checkpoint-f-level3-first-action.png` | `checkpoint-f-level3-aha.png` | `checkpoint-f-level3-completion.png` |
| Level 8 · When the ruler moves | `checkpoint-f-level8-goal.png` | `checkpoint-f-level8-first-action.png` | `checkpoint-f-level8-aha.png` | `checkpoint-f-level8-completion.png` |
| Level 9 · Build the fast attention path | `checkpoint-f-level9-goal.png` | `checkpoint-f-level9-first-action.png` | `checkpoint-f-level9-aha.png` | `checkpoint-f-level9-completion.png` |

The first-action images intentionally include productive failures for Levels 3, 8, and 9. The aha images show the newly understood mechanism: a direct shortcut, Welford’s auditable three-slot backpack, the online-softmax rescaling equation, and removal of attention’s two N×N parcels.

The corresponding browser path is asserted by `tests/ncd-game.mjs`; it also regenerates these images. See `SELF-PLAYTEST.md` for the first-pass confusions and the revisions visible here.
