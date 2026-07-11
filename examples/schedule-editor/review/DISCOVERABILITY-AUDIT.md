# Discoverability audit — before fixes

Date: 2026-07-11  
Build: production `pnpm build` from `proto/ncd-game-3` at `916b6a61`, served with `vite preview`  
Persona/input: fresh Chromium context, 1440×1000, ordinary mouse movement/down/up and visible button clicks only. No `dispatchEvent`, DOM mutation, state API, or app-internal calls.

I completed the ten-second opening by dragging the visible parcel to the visible shelf, revealed the shorthand, and entered the lesson map. For each later lesson I clicked its visible map card, stopped immediately, captured the viewport, and answered using only the pixels in that capture.

## Level 0 — the long way home

Screenshot: `audit-before-level0.png`

**What tells me what to do?** The sentence says “Drag the glowing parcel onto KEEP NEARBY.” It supplies a verb, a source description, and the target’s exact visible label. The orange `A` parcel is isolated and glowing. The target is a large dashed region, contains an orange shelf, and repeats `KEEP NEARBY` inside it.

**Why would I act?** The blue dotted route visibly takes an absurd detour through slow memory while a dark direct route passes through the shelf. The 64 MB counter supplies a consequence.

**Verdict:** Discoverable within seconds. This is the only screen where the instruction, source affordance, target affordance, and physical reason all agree.

## Level 1 — Stop the round trips

Screenshot: `audit-before-level1.png`

**What tells me what to do?** A low-contrast sentence below the entire machine says “Touch a blue delivery gate to fold it into a warm shortcut.” Two blue dashed rectangles could be the gates.

**Why is that insufficient?** The rectangles look like passive diagram groups: they have no action verb, button surface, pointer/hand glyph, motion, or attached callout. Their orange `A/B` parcels resemble labels, not handles. “Touch” is vague on desktop, and the prompt neither points to a specific first gate nor says “click.” The attention hierarchy favors the large static operation cards and the score, while the only instruction is the smallest text on the screen.

**Verdict:** A handler is guessable, but the affordance is not. A player can reasonably scan this as an explanatory diagram with nothing to operate. This fails the within-seconds requirement.

## Level 3 — A useful backpack

Screenshot: `audit-before-level3.png`

**What tells me what to do?** The orange filled control says “Try one continuous pass →.” It looks like a button, uses an imperative verb, and is the only warm control in the scene.

**Why is it still compromised?** The lesson opened with the scroll position inherited from clicking its lower map card. The screenshot is visibly clipped: global navigation and portions of the score card are above the viewport. The prompt in the blue strip says “Try to make the conveyor keep moving,” but does not explicitly say to click the orange control. A player should still find this particular action, but the broken entry position makes the page feel mounted mid-state.

**Verdict:** The first button itself is discoverable; lesson entry is not reliable. This does not support the user’s report by itself, but it confirms a progression/mount defect that worsens lower lessons and was invisible to selector-driven tests.

## Level 8 — When the ruler moves

Screenshot: `audit-before-level8.png`

**What tells me what to do?** The orange filled control says “Try one continuous pass →.” As in Level 3, it is the only control-looking element and names an action.

**Why might a player still hesitate?** “Try one continuous pass” is a prediction button rather than direct manipulation; nothing in the row illustration moves or invites touch. The blue strip describes a desired outcome, not the gesture. However, the filled button is sufficiently conventional to click.

**Verdict:** Initial action discoverable, though not as tactile as Level 0. Later steps need the same explicit guidance contract; the current screen has no reusable guidance layer, only authored button copy.

## Level 9 — Build the fast attention path

Screenshot: `audit-before-level9.png`

**What tells me what to do?** The right panel says “Apply each idea to the bottleneck it describes” and lists five techniques.

**Why is that insufficient?** The five interactive rows look exactly like a static legend: transparent background, thin separators, muted icons, no button boundary, no “click” language, no hover state visible by look, and no marked first action. The mission explicitly allows several concepts but gives no concrete starting move. The huge score/probability parcels look more clickable than the actual controls, yet clicking them does nothing. The retained map scroll is worst here: the global header and parts of the mission/score are clipped on entry.

**Verdict:** Fails decisively. There is no visible distinction between controls and explanatory text, and no named first move. Tests can click these rows by accessible names; a human has no reason to believe they are buttons.

## Before-fix conclusion

The user’s verdict is justified. Level 0 has a coherent discoverability grammar that later lessons do not reuse. Level 1 hides interaction inside diagram styling; Levels 3/8 happen to use conventional buttons but lack a structural guidance layer; Level 9 renders its controls as a legend. Additionally, lesson entry preserves the lesson-map scroll offset, so lower lessons mount partially scrolled/clipped.

The central failure is not missing handlers. It is that interactive semantics exist in Svelte and Playwright locators but are not encoded consistently in the visual surface.

## Root-cause check against the four suspects

### (a) Tests exercised handlers, not discoverability — confirmed, primary

The old browser test located actions by hidden implementation knowledge: `data-testid="boundary-a"`, exact accessible names, and predetermined move order. Several LayerNorm/softmax clicks used `{ force: true }`, bypassing Playwright’s own visibility/stability checks. The test also disabled every animation and transition before playing. It never asserted that a prompt named its target, that controls shared a visual affordance, or that the supposed next action was visually distinguished from static ink. The passing result proved that known handlers could mutate state; it did not prove that a person could find them.

### (b) Guidance initialized only for Level 0 — architectural absence, not conditional mount

There was no reusable guidance layer to initialize for lessons 1–9. Level 0 had its own explicit source/gesture/target sentence. The later component had a generic `feedback` string and hand-written button copy, but no data structure binding “next instruction” to “visible target.” Thus nothing failed to mount behind a flag; the required layer simply did not exist outside Level 0.

### (c) Progression/disabled state — not the first-action cause; one related state bug confirmed

All intended first controls were enabled. Later build steps were deliberately disabled until prior semantic moves, and their handlers worked. However, map and lesson views reused the same fixed scrolling element. `startLevel()` changed content without resetting `scrollTop`, so clicking lower map cards could open later lessons at the map’s retained offset. That made lesson entry appear clipped or mid-state and compounded the affordance problem.

### (d) Served build drift — ruled out

The HTTP document served on `127.0.0.1:5178` referenced `index-NCPfZIj8.js` and `index-BhvUX60T.css`; the production `dist/index.html` built from the audit commit referenced the same asset hashes. The unusable surface was the code we tested, not a stale or different bundle.

## Actual causes

1. No cross-lesson guidance/action contract.
2. Interactive panels styled like explanatory diagram ink, especially Level 1 gates and the capstone tool list.
3. Tests encoded secret target knowledge and sometimes forced clicks.
4. Shared scroll state was retained when switching from the map to a lesson.
