# Self-playtest: NCD learning game rebuild

Date: 2026-07-11

Persona: I played as a programmer who can read array code and recognizes softmax, but does not know GPU memory hierarchies, kernel fusion terminology, NCD notation, Welford’s algorithm, or FlashAttention’s implementation. I knew the implementation while testing, so I deliberately narrated only what the current screen made available and treated any move I remembered from the code as unavailable until the interface motivated it.

Method: clean browser load at 1440×1000, mouse only for the first pass, no test selectors or console. I completed level 0 and all four lessons in order, opened the paper shorthand only after each completion, returned to the map between lessons, and then replayed the changed build after the revisions below.

## First pass, minute by minute

### 0:00–0:25 — the long way home

The glowing parcel and breathing warm shelf won the attention contest immediately. I dragged before reading the traffic subtitle. The route folded, 64 MB became 48 MB, and “one write and one read” made the number change causal rather than congratulatory. Revealing the warm paper segment felt like a label for the shortcut I had already made.

Confusion: “Open freeform sandbox” was visible in the header on second one. As this persona, I did not know what “freeform” or the sandbox contained, and it competed with the only action the lesson wanted. It existed solely to make an old interaction test convenient.

Revision: removed that control from the opening. The advanced sandbox now becomes available on the lesson map, after the ten-second experiment. The canvas interaction test earns its way there through the intro instead of weakening the intro for the test.

### 0:25–1:20 — the elementwise chain

The two blue delivery gates were clearly repetitions of the first interaction. After the first click, the feedback “16 MB less” matched the 64→48 change. After the second, the completion sentence finally named fusion. This was the first moment I could define fusion without memorizing it: adjacent work shares the temporary parcel instead of shipping it away.

Confusion: the title “Stop shipping the glue” was playful but made me ask what the glue was. The `Hₗ₁/Mₗ₁` footer was also present before the lesson had earned either symbol, recreating a small version of the jargon-first problem.

Revision: renamed the level “Stop the round trips.” Hid the notation ledger during the first attempt and reveal it with completion. Later levels retain it as spaced retrieval of a label already introduced.

### 1:20–3:10 — LayerNorm

“Try one continuous pass” felt like my idea, not a deliberately illegal gesture. The machine then showed the mean changing from 3 to 6 at the variance station; I understood the obstruction before seeing Welford. Feeding `[2,4]` and `[8,10]` made the three-slot backpack concrete. Installing it after the experiment felt earned.

Confusion: the backpack jumped from `M2=2` to `M2=40`. I could believe the number, but could not reconstruct it. That made one slot feel like residual magic even though the card mechanism was gone.

Revision: after the second block, the lab now writes `(−4)² + (−2)² + 2² + 4² = 40` and calls M2 the running sum of squared distances. It is enough proof for the four-number witness without turning the screen into a derivation lecture.

### 3:10–5:00 — softmax

This was the strongest teaching beat. The bar at 4 physically moved the ruler beyond the earlier max 2. I deliberately chose “keep 1.368 and add”; the interface kept the wrong 2.368 visible and explained that the old bars were measured from a different maximum. The repair `1.368 × exp(2−4) + 1` then had a job to do. When `(m,ℓ)` appeared, both symbols referred to objects I had manipulated: ruler and repaired subtotal.

No blocking confusion. I did initially read `ℓ` as a decorative letter, but the adjacent word “subtotal” resolved it. Keeping both is useful.

### 5:00–7:20 — attention capstone

The two N×N parcels were legible bottlenecks, and the tool descriptions recalled previous experiences rather than introducing fresh verbs. I tried flowing key columns before installing `(m,ℓ)`; the feedback referred back to the moving-ruler lesson, so the failure felt diagnostic. Removing the two giant parcels produced the visual payoff the earlier game lacked.

Confusion: “Work on 64 query rows” said it would fit the nearby workbench, but the score card displayed only traffic. I had no evidence that query tiling was needed; it felt like a fifth checkbox.

Revision: every level with an `M` target now shows **nearby workbench** and **space limit** in physical units alongside traffic. Query tiling therefore has a visible resource to improve. The compact `Mₗ₁` label remains secondary.

## Second pass after revisions

I replayed from a clean load and specifically checked each earlier hesitation:

- At second one there are only the animated experiment and an escape back to the schedule editor. I acted without choosing a mode.
- Level 1 used “round trips,” the phrase already explained by the intro. No `Hₗ₁/Mₗ₁` appeared until completion.
- I could account for every Welford backpack value in the four-number example, including 40.
- The softmax wrong path remained recoverable and explanatory; no revision was needed.
- Before touching a capstone tool I could state two independent problems: 54 MB of traffic and a 24 MB nearby footprint versus a much smaller limit. The tiling action now answered the second problem.

The remaining limitation is representational, not a hidden UI rule: the small LayerNorm/softmax teaching terms model row-sized intermediates where a production kernel has scalar carried state plus a row program (finding F30). The game calls the three/two scalar slots out explicitly, but the paper projection cannot yet derive that distinction from the term.

## Verdict

This pass is pedagogically coherent enough to hand to a real first-time tester. The first action needs no domain vocabulary; each later technical name compresses an experience; the two lemmas are necessary repairs demonstrated on numbers; and the capstone asks for composition rather than recollection of an arbitrary gesture order. It is still a short authored lesson, not proof that the interaction vocabulary generalizes to arbitrary schedules. A human study should next test whether players can explain the online correction and diagnose a new, near-neighbor streaming obstruction without prompts.

## Discoverability addendum — lessons 1+ (`proto/ncd-game-3`)

Trigger: direct user report that only the opening lesson worked and every later lesson lacked an obvious action.

I replayed lessons 1, 3, 8, and 9 from a fresh production-preview context using ordinary pointer input and, on the first pass, stopped at each entry screen without consulting the DOM. The before verdicts are recorded with screenshots in `DISCOVERABILITY-AUDIT.md`. The short version: Level 1’s gates looked like passive diagram groups; Levels 3/8 had conventional orange buttons but no common guidance system; Level 9’s actual buttons looked like a static legend. Lower map entries also retained scroll position.

After the fix I replayed the full path while following only the visible yellow “YOUR NEXT MOVE” panel and orange target halo:

- **Level 1:** the panel named the blue `A` gate and its exact location. Both gates now carry `CLICK` badges and button surfaces. After A, the panel changed to B without covering the cost feedback.
- **Level 3:** the entry panel named the orange continuous-pass button. Each state transition then named the four-number experiment, current feed button, Welford install, nearby step, and stream step. I never had to scan for a newly appeared control.
- **Level 8:** the same grammar survived a branch. At the subtotal choice it recommended the pedagogically correct rescale path while leaving the deliberately wrong option available.
- **Level 9:** the panel named one valid first action—Carry `m,ℓ`—and that tool was moved to the top of a visibly button-like stack. Subsequent guidance followed the same learned order. The capstone no longer asks the player to infer that a legend is clickable.

The map/lesson scroll position reset to zero on every entry, including the bottom capstone card. I also tightened the first-pass scenes so their highlighted button remains visible at ordinary laptop heights.

Self-critique after replay: this version is intentionally more directive. That is the right correction for a surface whose controls were undiscoverable, but the guidance should eventually become collapsible after a player demonstrates fluency. For v1 it remains persistent because reliable action discovery matters more than preserving puzzle ambiguity.
