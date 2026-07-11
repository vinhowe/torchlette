# NCD game rebuild: teaching design

Status: Commit A design authority document. Section 5 of `docs/ncd-surface-spec.md` is background, not a constraint, for this game surface.

## Teaching thesis

The player should meet a wasteful computation before they meet its vocabulary.

The old game began with a level card, a budget, a notation-heavy canvas, and a palette of legal moves. That is an editor tutorial disguised as a game. It asks the player to translate unfamiliar symbols into mechanics before they have a reason to care about either. The rebuild reverses that order:

1. Show a physical process with one visible inconvenience.
2. Let the player change the process immediately.
3. Animate the consequence and interpret it in ordinary language.
4. Give the action a short reusable name.
5. Reveal the corresponding NCD mark as earned compression.
6. Later, ask the player to recognize and compose the same idea without the physical scaffold.

The primary feedback is a sentence about the machine: “That temporary result stayed nearby, so 16 MB never made the round trip to slow memory.” The number, animation, and diagram all corroborate that sentence. A changed cost cell is supporting evidence, not the lesson.

## Registers studied and what carries forward

- **Nicky Case explorables:** LOOPY explicitly frames learning as “messing around and seeing what happens,” while Case’s teaching notes recommend purpose, intuition, then feedback-rich practice. The opening therefore begins with a movable object in a running system, not prose or notation. The interface is the explanation, and optional detail appears in context rather than in a glossary. Sources: [LOOPY](https://ncase.me/loopy/), [Nicky Case learning notes](https://ncase.me/faq/), [Nutshell’s just-in-time explanation pattern](https://ncase.me/nutshell/).
- **The Witness:** a curated sequence lets the player infer a small rule, then varies one condition at a time. Each lesson here has one “sentence”: keep a value nearby; carry enough state to continue; repair a changing reference point; compose the learned moves. The game does not name a rule before the player has demonstrated it. Jonathan Blow describes puzzles as being about an idea rather than merely a thing to solve; that is the bar for each level. Source: [Jonathan Blow interview](https://time.com/4355763/the-witness-jonathan-blow-interview/).
- **Zachtronics:** the target is an inspectable machine, not a multiple-choice answer. Solutions remain concrete constructions with history and replay. The player should be able to point at what they changed and why it works. We borrow the pleasure of making a small machine run, but avoid dropping a manual on the player before the first action. Reference: [Open-Ended Puzzle Design at Zachtronics](https://www.classcentral.com/course/youtube-open-ended-puzzle-design-at-zachtronics-165884).
- **Factorio’s first hour:** goals are expressed as outcomes in the world, and a working automation gives immediate, legible payoff. Factorio’s own new-player analysis warns that heavily constrained actions can teach the tutorial solution rather than the underlying concept. Our early scenes are curated, but the object being manipulated is the same dependency and residency structure used later. Sources: [Factorio new-player experience](https://direct.factorio.com/blog/post/fff-241), [Factorio tutorial structure](https://wiki.factorio.com/Tutorial).
- **Brilliant:** one concept per lesson, concrete computation, hands-on manipulation, and immediate tailored feedback. We use short prediction/try/interpret cycles and remove scaffolding at the capstone. We do not copy streaks, XP, or decorative gamification; the content has to supply the reward. Source: [Brilliant’s learning method](https://brilliant.org/about/).

## Player journey, minute by minute

### Second 1

The player sees a running three-station computation. Bright parcels flow left to right. One parcel takes a conspicuous U-shaped detour through a distant, cold-blue “slow memory” depot. A large counter says “64 MB moved this step.” The only animated affordance is the parcel; the nearby landing zone breathes gently. The six-word prompt is: **Drag the parcel to KEEP NEARBY.**

There is no level select, formula, axis label, palette, score, or NCD acronym.

### Seconds 2–10

The player drags the glowing parcel a short distance. The long route folds into a short bridge. The counter rolls from 64 MB to 48 MB. The courier animation stops visiting the depot. Feedback appears beside the changed route:

> You removed one round trip. This temporary result stays close, so 16 MB no longer travels to slow memory.

The action is reversible. Reset is visible after success, not before.

### Seconds 10–35

The game overlays a tiny piece of NCD shorthand on the same route. The highlighted local segment receives the warm lower-memory color and the intermediate’s axis stack. Copy says: “This color is just shorthand for the promise you made: keep this value nearby.” The player may flip between “machine” and “shorthand” views; the geometry remains aligned so the mapping is not a lecture.

### 35 seconds–2 minutes

Level 1 presents the full bias → GELU → residual chain with two identical detours. The player repeats the known action without an arrow pointing to the exact answer. The second success reveals the word **fuse**: adjacent work shares a nearby intermediate instead of writing it out and reading it back. The cost ledger shows both concrete traffic and the compact paper symbol `Hₗ₁`, with the plain name always first.

At two minutes the player should understand:

- operations can be cheap while moving their intermediates is expensive;
- keeping an intermediate near the compute removes a write and a read;
- the warm-colored NCD region records that decision;
- `Hₗ₁` is a compact count of that movement, not an arbitrary score.

They should not yet need to know “global memory,” “island,” “group partition,” or “streamability.”

## Lesson architecture

Every level uses a four-beat loop:

1. **Notice:** an animated system makes the pain observable.
2. **Predict:** the player chooses or performs a concrete intervention.
3. **Run:** the machine visibly executes the new plan.
4. **Name:** feedback interprets the physical consequence, then reveals the reusable term or notation.

Failure is diagnostic and local. It changes the machine enough to expose the dependency. It does not merely reject a drop.

### Level 0 — The long way home

- **One concept:** an intermediate can stay nearby instead of making a round trip.
- **Felt before named:** a parcel visibly travels to a distant depot and back.
- **Allowed failure:** dropping anywhere else makes the parcel spring back; the depot briefly traces the route it would still take and says “That still sends it away.”
- **Getting-it moment:** the route collapses, traffic falls by 16 MB, and the warm NCD segment appears over the shortcut.
- **Vocabulary earned:** near memory, round trip, one lower-level color.

### Level 1 — Stop the round trips

- **One concept:** fuse a chain of elementwise operations by keeping both intermediates nearby.
- **Felt before named:** the arithmetic stations finish almost instantly, but courier trips dominate the clock and traffic.
- **Allowed failure:** joining non-neighboring stations leaves the middle intermediate’s depot trip glowing; feedback says exactly which value still escapes.
- **Getting-it moment:** all three stations pulse as one local workshop; two detours disappear; the player hears/reads “two writes + two reads removed.”
- **Vocabulary earned:** fuse, boundary, `Hₗ₁` as shorthand for traffic crossing the warm-memory boundary.

### Level 3 — A useful backpack

- **One concept:** a pass can continue block by block if it carries a sufficient summary of the past.
- **Felt before named:** the player feeds row chunks through mean → variance → normalize. A naive one-pass attempt reaches variance, which visibly asks for a mean that is still changing. The conveyor pauses on the exact dependency.
- **Allowed failure:** the player is encouraged to press “send the next chunk” early. This is a productive prediction, not an illegal gesture. A small numeric row shows why the current mean is provisional.
- **Discovery:** an interactive backpack lets the player retain `(count, mean, M2)`. They step two tiny blocks and see the three values update. Only after this concrete rehearsal does the card get the name “Welford running moments.” Applying it is therefore not magic.
- **Getting-it moment:** chunks flow continuously while the backpack travels with them; full rows stop being parked in slow memory.
- **Vocabulary earned:** carry state, stream, the stream mark, Welford.

### Level 8 — When the ruler moves

- **One concept:** online softmax must repair old contributions when a later block raises the maximum.
- **Felt before named:** the player accumulates exponentials for a first block, then reveals a larger number in the second. The old bars visibly shrink relative to the new maximum; the previously computed sum is now in the wrong scale.
- **Allowed failure:** “just keep adding” produces a visibly wrong normalized total. The game overlays the correct bars rather than announcing an abstract streamability violation.
- **Discovery:** the player chooses “rescale what I already have.” A slider moves the maximum and the old subtotal scales by `exp(old max − new max)`. Only after the player sees the repair does `(m, ℓ)` appear as the compact carried state.
- **Getting-it moment:** a second example changes the maximum again and the player predicts which subtotal must shrink.
- **Vocabulary earned:** running maximum `m`, running normalizer `ℓ`, online-softmax correction.

### Level 9 — Build the fast attention path

- **One concept:** FlashAttention is composition of already-understood traffic, chunking, and carried-state ideas.
- **Felt before named:** the eager attention machine shows two enormous square parcels occupying the depot. Hovering either answers “what is this?” in plain language: all pairwise scores or all normalized probabilities.
- **Allowed failure:** fusing before choosing manageable chunks overfills the nearby workbench; streaming before installing the learned `(m, ℓ)` state replays the familiar moving-ruler counterexample. No new rule is introduced.
- **Getting-it moment:** the player assembles the path from their earned tools; the square intermediates fade from slow memory, the machine runs chunk by chunk, and a full NCD diagram replaces the scaffold without changing structure.
- **Vocabulary earned:** group/tile labels and the full paper notation as a compressed record of their construction.

## Information and interaction design

### The machine view

The initial visual language is a workshop:

- operation stations perform named work;
- parcels are arrays/intermediates;
- the distant blue depot is slow memory;
- the warm workbench is limited nearby memory;
- route thickness and moving parcels communicate transfer volume;
- nearby capacity is shown as occupied physical slots before it is shown as `Mₗ₁`.

The mapping to NCD is exact rather than metaphorical decoration: parcels map to wires, stations to boxes, route crossings to color changes, the workbench capacity to maximum column width, and traffic to color-changing wire size. The semantic `NcdTerm` remains the source of truth.

### Feedback grammar

Every consequential action emits all three layers, in this order:

1. **What happened:** “The score tensor stayed on the workbench.”
2. **Why it matters:** “That removes a write and a read of 6.0 MB.”
3. **How the paper records it:** “The wire remains warm-colored across this boundary.”

If a move fails:

1. show the attempted machine state;
2. animate the exact missing dependency or exceeded capacity;
3. ask a concrete question;
4. offer optional hints only after the player has inspected the failure.

There are no generic “illegal,” “streamability,” or “target not met” messages in the teaching path.

### Numbers

The game states the workload behind every displayed byte count. Early lessons use the semantic term’s per-row napkin count multiplied by a visible mini-batch row count. Later attention lessons use tensor extents directly. Compact symbols are paired with names:

- `memory traffic (Hₗ₁)`
- `nearby space needed (Mₗ₁)`

The plain phrase is never removed.

## What this design overrules from Section 5

- **No cold level select.** The first launch enters level 0 mid-motion. A map appears after the first success and remains reachable thereafter.
- **No stated goal before the first diagram.** The opening goal is embodied by an obvious detour and a six-word action prompt. Later levels use short missions after the player has a vocabulary.
- **No palette-first control scheme.** Early actions are direct manipulation of the machine. Reusable tools become a palette only after the player has earned and understood them.
- **No jam-to-unlock lemma gate.** A lemma is learned through a tiny manipulable counterexample, then named. Failure still creates the need, but the card is not a reward dispensed by a refusal flag.
- **No cost table as the primary score.** Concrete traffic, space, route time, and visible objects lead. The aligned napkin table arrives as a compact audit view.
- **No hidden “derive FA” or autoplay.** The capstone is player-built. Replays may explain the player’s own completed history, never substitute for the construction.
- **No Sequence UI in the game.** The game uses its own scale, motion, illustration, and responsive layout. The advanced sandbox can retain the instrument/editor chrome because it serves a different audience and purpose.

## Instrumentation and playtest questions

Mechanical assertions remain necessary: actions mutate the semantic term, costs come from `napkinCost`, undo is inverse relabeling, and final semantic hashes match known solutions. The self-playtest must additionally answer:

- Was the first draggable object recognized without reading?
- Did the player predict the direction of the byte-count change?
- Can they explain “fuse” without using the word?
- Does Welford feel like a necessary carried summary rather than a magic card?
- Can they explain the online-softmax correction by pointing to the moving maximum?
- In the capstone, do they choose tools based on visible problems rather than remembered UI choreography?
