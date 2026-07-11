<script lang="ts">
  import { tick } from "svelte";
  import NcdGame from "./NcdGame.svelte";
  import NcdRenderer from "./NcdRenderer.svelte";
  import NcdView from "./NcdView.svelte";
  import { FA_STEPS } from "../../ncd/fa-script";
  import {
    GAME_LEVELS,
    levelById,
    softmaxGameLemma,
    welfordLemma,
    type GameLevelId,
  } from "../../ncd/game-levels";
  import {
    applyMove,
    cloneTerm,
    inverseMove,
    napkinCost,
  } from "../../ncd/model";
  import type { NcdMove, NcdTerm, PartitionKind, NcdLevel } from "../../ncd/types";

  let { onExit }: { onExit: () => void } = $props();

  type Screen = "intro" | "map" | "lesson" | "sandbox";
  type LessonPhase = "notice" | "failure" | "lab" | "build" | "complete";
  type HistoryGroup = { label: string; forward: NcdMove[]; inverse: NcdMove[] };
  type Completion = { traffic: number; nearby: number; actions: number };
  type LessonGuidance = { actionId: string; instruction: string; why: string };

  let learningGameElement = $state<HTMLElement>();
  let screen = $state<Screen>("intro");
  let selectedId = $state<GameLevelId | null>(null);
  let term = $state<NcdTerm | null>(null);
  let phase = $state<LessonPhase>("notice");
  let feedback = $state("Touch the machine and watch what changes.");
  let history = $state<HistoryGroup[]>([]);
  let actionCount = $state(0);
  let progress = $state<Partial<Record<GameLevelId, Completion>>>({});
  let labStep = $state(0);
  let softmaxChoice = $state<"none" | "wrong" | "rescale">("none");
  let notationOpen = $state(false);
  let capstoneMoves = $state<string[]>([]);
  let blockedTool = $state<string | null>(null);

  const level = $derived(selectedId ? levelById(selectedId) : null);
  const cost = $derived(term ? napkinCost(term) : null);
  const targetTrafficBytes = $derived(
    level ? level.target.h * 4 * workloadScale(level.id) : 0,
  );
  const targetNearbyBytes = $derived(
    level?.target.m ? level.target.m * 4 : 0,
  );
  const guidance = $derived.by((): LessonGuidance | null => {
    if (!level) return null;
    if (phase === "complete") {
      return {
        actionId: "lesson-map",
        instruction: "Click “Back to lesson map” when you are ready.",
        why: "This machine is complete; the map records your result.",
      };
    }
    if (level.id === "fuse-chain") {
      return hasLocal("mid1", 2)
        ? {
            actionId: "fuse-boundary-b",
            instruction: "Click the glowing blue B gate between GELU and + residual.",
            why: "B is the remaining 16 MB round trip.",
          }
        : {
            actionId: "fuse-boundary-a",
            instruction: "Click the glowing blue A gate between + bias and GELU.",
            why: "This folds A’s write-and-read detour into a nearby shortcut.",
          };
    }
    if (level.id === "layernorm") {
      if (phase === "notice") return { actionId: "layernorm-try-flow", instruction: "Click the orange “Try one continuous pass” button below the row.", why: "Run the natural idea first so the exact dependency becomes visible." };
      if (phase === "failure") return { actionId: "layernorm-open-lab", instruction: "Click “Open the 4-number experiment.”", why: "The small example will show what variance must remember." };
      if (phase === "lab" && labStep === 0) return { actionId: "welford-feed", instruction: "Click the orange button to feed [2, 4] into the backpack.", why: "Watch which three numbers summarize the block." };
      if (phase === "lab" && labStep === 1) return { actionId: "welford-feed", instruction: "Click the orange button again to feed [8, 10].", why: "The same three slots should summarize all four values." };
      if (phase === "lab") return { actionId: "welford-install", instruction: "Click “Carry this backpack through LayerNorm.”", why: "Install the running summary you just verified." };
      if (!hasLocal("mid1", 2)) return { actionId: "layernorm-keep-nearby", instruction: "Click step 1: “Keep each chunk nearby.”", why: "This removes both whole-row round trips." };
      return { actionId: "layernorm-stream", instruction: "Click step 2: “Flow 128 values at a time.”", why: "The Welford backpack can now travel between blocks." };
    }
    if (level.id === "softmax") {
      if (phase === "notice") return { actionId: "softmax-try-flow", instruction: "Click the orange “Try one continuous pass” button below the row.", why: "Run the obvious idea first and watch where its scale breaks." };
      if (phase === "failure") return { actionId: "softmax-open-lab", instruction: "Click “Try it with [2, 1] then [4].”", why: "Three scores are enough to see the maximum move." };
      if (phase === "lab" && labStep < 2) return { actionId: "softmax-reveal", instruction: `Click the orange button to reveal the ${labStep === 0 ? "first" : "next"} block.`, why: "Compare each subtotal with the maximum used as its ruler." };
      if (phase === "lab" && softmaxChoice === "none") return { actionId: "softmax-rescale", instruction: "Click “Shrink it to the new ruler.”", why: "The earlier subtotal was measured against max 2, not max 4." };
      if (phase === "lab" && softmaxChoice === "wrong") return { actionId: "softmax-rescale", instruction: "Click “Rescale the old subtotal.”", why: "Repair the deliberately wrong 2.368 result." };
      if (phase === "lab") return { actionId: "softmax-install", instruction: "Click “Carry the ruler m and subtotal ℓ.”", why: "Install the two values whose update you just checked." };
      if (!hasLocal("mid1", 2)) return { actionId: "softmax-keep-nearby", instruction: "Click step 1: “Keep score chunks nearby.”", why: "This removes both full-row round trips." };
      return { actionId: "softmax-stream", instruction: "Click step 2: “Flow 128 scores at a time.”", why: "The m,ℓ state can now repair each arriving block." };
    }
    if (!faDone("admit-online-softmax")) return { actionId: "fa-carry", instruction: "Start by clicking the orange “Carry m and ℓ” tool on the right.", why: "You already proved that softmax needs this state before key blocks can flow." };
    if (!faDone("fuse-qk-softmax")) return { actionId: "fa-scores", instruction: "Click “Keep score blocks nearby.”", why: "This removes the first giant N×N parcel from slow memory." };
    if (!faDone("fuse-softmax-pv")) return { actionId: "fa-probabilities", instruction: "Click “Keep probability blocks nearby.”", why: "This removes the second giant N×N parcel." };
    if (!faDone("tile-q")) return { actionId: "fa-tile", instruction: "Click “Work on 64 query rows.”", why: "Smaller query groups fit the nearby-workbench limit." };
    return { actionId: "fa-stream", instruction: "Click “Flow 32 key columns.”", why: "Finish the schedule by moving keys and values through the carried state." };
  });

  function workloadScale(id: GameLevelId): number {
    return id === "attention" ? 1 : 2048;
  }

  function trafficBytes(value: NcdTerm, id: GameLevelId): number {
    return napkinCost(value).transferBytesByLevel.l1 * workloadScale(id);
  }

  function formatBytes(bytes: number): string {
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(bytes >= 10 * 1024 * 1024 ? 0 : 1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(bytes >= 10 * 1024 ? 0 : 1)} KB`;
    return `${bytes} B`;
  }

  async function resetView(): Promise<void> {
    await tick();
    learningGameElement?.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }

  async function showMap(): Promise<void> {
    screen = "map";
    await resetView();
  }

  async function startLevel(id: GameLevelId): Promise<void> {
    const next = levelById(id);
    selectedId = id;
    term = cloneTerm(next.baseline);
    phase = "notice";
    feedback = openingFeedback(id);
    history = [];
    actionCount = 0;
    labStep = 0;
    softmaxChoice = "none";
    notationOpen = false;
    capstoneMoves = [];
    blockedTool = null;
    screen = "lesson";
    await resetView();
  }

  function openingFeedback(id: GameLevelId): string {
    if (id === "fuse-chain") return "The arithmetic is fast. Watch the two temporary parcels make the long trip.";
    if (id === "layernorm") return "Three passes cross the same row. Try to make the conveyor keep moving.";
    if (id === "softmax") return "Softmax rereads a whole row three times. See what stops a single flowing pass.";
    return "Two square intermediates dominate the depot. Reuse the four ideas you already learned.";
  }

  function applyGroup(label: string, moves: NcdMove[], message: string): void {
    if (!term || !level) return;
    let next = term;
    const inverses: NcdMove[] = [];
    for (const move of moves) {
      next = applyMove(next, move);
      inverses.unshift(inverseMove(move));
    }
    term = next;
    history = [...history, { label, forward: moves, inverse: inverses }];
    actionCount += 1;
    feedback = message;
    blockedTool = null;
    maybeComplete(next);
  }

  function undo(): void {
    if (!term || !history.length || phase === "complete") return;
    const entry = history.at(-1);
    if (!entry) return;
    let next = term;
    for (const move of entry.inverse) next = applyMove(next, move);
    term = next;
    history = history.slice(0, -1);
    actionCount = Math.max(0, actionCount - 1);
    feedback = `Undid “${entry.label}.” The machine is back to its previous route.`;
  }

  function maybeComplete(next: NcdTerm): void {
    if (!level) return;
    const nextCost = napkinCost(next);
    const met =
      nextCost.transferByLevel.l1 <= level.target.h &&
      (level.target.m === undefined || nextCost.memoryByLevel.l1 <= level.target.m);
    if (!met) return;
    phase = "complete";
    progress = {
      ...progress,
      [level.id]: {
        traffic: trafficBytes(next, level.id),
        nearby: nextCost.memoryBytesByLevel.l1,
        actions: actionCount,
      },
    };
  }

  function fuseChainBoundary(wireId: "mid1" | "mid2", column: 2 | 4): void {
    const label = wireId === "mid1" ? "A" : "B";
    applyGroup(
      `keep ${label} nearby`,
      [{ op: "recolor", wireId, column, before: "l0", after: "l1" }],
      `The ${label} parcel now takes the short bridge. One write and one read disappeared: 16 MB less traffic this step.`,
    );
  }

  function hasLocal(wireId: string, column: number): boolean {
    return Boolean(
      term?.decorations.residency.some(
        (item) => item.wireId === wireId && item.column === column && item.level === "l1",
      ),
    );
  }

  function tryLayernormFlow(): void {
    phase = "failure";
    actionCount += 1;
    feedback = "The conveyor paused at variance: its mean is still changing, so the next station does not yet know what to subtract.";
  }

  function openWelfordLab(): void {
    phase = "lab";
    labStep = 0;
    feedback = "Feed two tiny blocks. Look for a summary small enough to travel with the conveyor.";
  }

  function feedMomentBlock(): void {
    labStep = Math.min(2, labStep + 1);
    actionCount += 1;
    feedback = labStep === 1
      ? "After [2, 4], three numbers summarize everything seen: count 2, mean 3, spread 2."
      : "After [8, 10], the same three slots now summarize all four values. The row itself never had to come back.";
  }

  function installWelford(): void {
    if (!term) return;
    applyGroup(
      "carry running moments",
      [welfordLemma(term)],
      "You packed count, mean, and M2 into a traveling summary. This is Welford’s running-moments rule.",
    );
    phase = "build";
  }

  function trySoftmaxFlow(): void {
    phase = "failure";
    actionCount += 1;
    feedback = "The second block contains a larger maximum. That changes the scale of every exponential already counted.";
  }

  function openSoftmaxLab(): void {
    phase = "lab";
    labStep = 0;
    softmaxChoice = "none";
    feedback = "Reveal the blocks one at a time and keep an eye on the ruler—the running maximum.";
  }

  function revealSoftmaxBlock(): void {
    labStep = Math.min(2, labStep + 1);
    actionCount += 1;
    feedback = labStep === 1
      ? "For [2, 1], the ruler is m = 2 and the exponential subtotal is ℓ = 1.368."
      : "A later score 4 moves the ruler from 2 to 4. The old subtotal is now measured in the wrong scale.";
  }

  function chooseOldSubtotal(): void {
    softmaxChoice = "wrong";
    actionCount += 1;
    feedback = "Just adding gives 2.368, but the old bars should shrink when the ruler moves. This would not normalize to the right probabilities.";
  }

  function chooseRescale(): void {
    softmaxChoice = "rescale";
    actionCount += 1;
    feedback = "Yes: multiply the old subtotal by exp(2−4) = 0.135, then add the new block. The repaired subtotal is 1.185.";
  }

  function installOnlineSoftmax(): void {
    if (!term) return;
    applyGroup(
      "carry m and ℓ",
      [softmaxGameLemma(term)],
      "You carry two numbers: m, the ruler, and ℓ, the subtotal expressed against that ruler. This is online softmax.",
    );
    phase = "build";
  }

  function keepPassesNearby(): void {
    applyGroup(
      "keep row chunks nearby",
      [
        { op: "recolor", wireId: "mid1", column: 2, before: "l0", after: "l1" },
        { op: "recolor", wireId: "mid2", column: 4, before: "l0", after: "l1" },
      ],
      "Both temporary chunks stay on the warm workbench. Two complete slow-memory round trips are gone.",
    );
  }

  function streamRows(): void {
    if (!term) return;
    applyGroup(
      "flow in blocks of 128",
      [{
        op: "partition",
        axisId: "r",
        before: term.decorations.partitions.find((item) => item.axisId === "r"),
        after: { axisId: "r", kind: "stream", size: 128, label: "s_r" },
      }],
      "The row now flows in 128-value blocks. Only one block and the carried summary need nearby space.",
    );
  }

  function hasRowStream(): boolean {
    return Boolean(term?.decorations.partitions.some((item) => item.axisId === "r" && item.kind === "stream"));
  }

  function performCapstone(stepIndex: number): void {
    if (!term) return;
    const step = FA_STEPS[stepIndex];
    if (!step || capstoneMoves.includes(step.id)) return;
    if (step.id === "stream-x" && !capstoneMoves.includes("admit-online-softmax")) {
      blockedTool = step.id;
      actionCount += 1;
      feedback = "The key blocks cannot flow yet: softmax still forgets how earlier values should change when a later maximum arrives. Reuse the m,ℓ backpack first.";
      return;
    }
    const move = step.makeMove(term);
    const interpretations: Record<string, string> = {
      "admit-online-softmax": "The familiar m,ℓ backpack is installed. Softmax can now continue when the next key block arrives.",
      "fuse-qk-softmax": "The full score square no longer goes to slow memory. Each score block feeds softmax nearby.",
      "fuse-softmax-pv": "The probability square also stays nearby. Each normalized block feeds the value product directly.",
      "tile-q": "The workshop handles 64 query rows at a time, keeping its nearby footprint within the available bench.",
      "stream-x": "Keys and values now flow through 32 columns at a time while m and ℓ carry the history.",
    };
    capstoneMoves = [...capstoneMoves, step.id];
    applyGroup(step.label, [move], interpretations[step.id] ?? step.label);
  }

  function faDone(id: string): boolean {
    return capstoneMoves.includes(id);
  }

  const noopPartition = (_axisId: string, _kind: PartitionKind): void => {};
  const noopResidency = (_wireId: string, _column: number, _level: NcdLevel): void => {};
  const noopLemma = (_boxId: string): void => {};
  const noop = (): void => {};
</script>

{#if screen === "intro"}
  <NcdGame {onExit} onComplete={showMap} />
{:else if screen === "sandbox"}
  <div class="sandbox-shell">
    <button class="sandbox-back" data-game-affordance="navigation" data-action-id="sandbox-back" onclick={showMap}>← Back to lessons</button>
    <NcdView startInSandbox />
  </div>
{:else}
  <main class="learning-game" data-testid="learning-game" bind:this={learningGameElement}>
    <header class="world-header">
      <button class="brand-button" data-game-affordance="navigation" data-action-id="brand-map" onclick={showMap}><span>↝</span> Memory Garden</button>
      <nav>
        <button class:active={screen === "map"} data-game-affordance="navigation" data-action-id="nav-map" onclick={showMap}>Lesson map</button>
        <button data-game-affordance="navigation" data-action-id="nav-sandbox" onclick={() => (screen = "sandbox")}>Advanced sandbox</button>
        <button data-game-affordance="navigation" data-action-id="nav-schedule" onclick={onExit}>Schedule editor</button>
      </nav>
    </header>

    {#if screen === "map"}
      <section class="lesson-map" data-testid="lesson-map">
        <div class="map-intro">
          <p class="eyebrow">Your route through the machine</p>
          <h1>Make the data travel less.</h1>
          <p>Each stop adds one idea. You will build FlashAttention from those ideas at the end.</p>
        </div>
        <div class="map-path">
          {#each GAME_LEVELS as item, index}
            <button class="map-card" class:complete={Boolean(progress[item.id])} data-game-affordance="navigation" data-action-id={`open-${item.id}`} onclick={() => startLevel(item.id)} aria-label={`Open ${item.title}`}>
              <span class="map-index">{index + 1}</span>
              <span class="map-copy">
                <small>{index === 0 ? "KEEP VALUES CLOSE" : index === 1 ? "CARRY A SUMMARY" : index === 2 ? "REPAIR THE SCALE" : "COMPOSE EVERYTHING"}</small>
                <strong>{item.id === "fuse-chain" ? "Stop the round trips" : item.id === "layernorm" ? "A useful backpack" : item.id === "softmax" ? "When the ruler moves" : "Build the fast attention path"}</strong>
                <em>{item.id === "fuse-chain" ? "bias → GELU → residual" : item.id === "layernorm" ? "LayerNorm, one flowing row" : item.id === "softmax" ? "softmax, without rereading" : "the capstone"}</em>
              </span>
              {#if progress[item.id]}
                <span class="map-result">✓ {formatBytes(progress[item.id]?.traffic ?? 0)} · {progress[item.id]?.actions} actions</span>
              {:else}
                <span class="map-arrow">→</span>
              {/if}
            </button>
          {/each}
        </div>
        <aside class="map-note">
          <strong>What you already discovered</strong>
          <span>A warm segment means a temporary value stayed near the compute.</span>
          <button data-game-affordance="navigation" data-action-id="replay-intro" onclick={() => (screen = "intro")}>Replay the 10-second experiment</button>
        </aside>
      </section>
    {:else if level && term && cost}
      <section class="lesson" data-level={level.id}>
        <div class="lesson-topline">
          <button class="back-button" data-game-affordance="navigation" data-action-id="lesson-map-top" onclick={showMap}>← Lesson map</button>
          <span>Lesson {GAME_LEVELS.findIndex((item) => item.id === level.id) + 1} of 4</span>
          <button class="undo-button" disabled={!history.length || phase === "complete"} onclick={undo}>Undo last change</button>
        </div>

        <header class="mission">
          <div>
            <p class="eyebrow">{level.id === "attention" ? "CAPSTONE" : "ONE NEW IDEA"}</p>
            <h1>{level.id === "fuse-chain" ? "Stop the round trips" : level.id === "layernorm" ? "A useful backpack" : level.id === "softmax" ? "When the ruler moves" : "Build the fast attention path"}</h1>
            <p>{level.id === "fuse-chain" ? "Three tiny operations are waiting on two long deliveries." : level.id === "layernorm" ? "Can LayerNorm cross a row once instead of three times?" : level.id === "softmax" ? "Can softmax keep a correct total when a later value changes the scale?" : "Remove the two giant square intermediates without overflowing the nearby workbench."}</p>
          </div>
          <div class="physical-score" class:won={phase === "complete"}>
            <div><span>Traffic this step</span><strong>{formatBytes(trafficBytes(term, level.id))}</strong></div>
            <div><span>Traffic goal</span><strong>{formatBytes(targetTrafficBytes)}</strong></div>
            {#if level.target.m !== undefined}
              <div class="space-row"><span>Nearby workbench</span><strong>{formatBytes(cost.memoryBytesByLevel.l1)}</strong></div>
              <div class="space-row"><span>Space limit</span><strong>{formatBytes(targetNearbyBytes)}</strong></div>
            {/if}
            <small>{phase === "complete" ? "Goal reached" : `${formatBytes(Math.max(0, trafficBytes(term, level.id) - targetTrafficBytes))} still removable`}</small>
          </div>
        </header>

        {#if guidance}
          <section class="action-guide" data-testid="lesson-guidance" data-target-action={guidance.actionId}>
            <span class="guide-kicker">YOUR NEXT MOVE</span>
            <div><strong>{guidance.instruction}</strong><p>{guidance.why}</p></div>
            <span class="guide-arrow">↘</span>
          </section>
        {/if}

        <div class="feedback" data-testid="teaching-feedback"><span>↳</span><p>{feedback}</p></div>

        {#if level.id === "fuse-chain"}
          <section class="chain-machine" data-testid="level-chain-machine">
            <div class="operation">+ bias<small>0.03 ms work</small></div>
            <button class="boundary" class:local={hasLocal("mid1", 2)} data-game-affordance="action" data-action-id="fuse-boundary-a" data-current-target={guidance?.actionId === "fuse-boundary-a"} disabled={hasLocal("mid1", 2)} onclick={() => fuseChainBoundary("mid1", 2)} data-testid="boundary-a">
              <span class="parcel-mini">A</span><strong>{hasLocal("mid1", 2) ? "stays nearby" : "send away + fetch back"}</strong><small>{hasLocal("mid1", 2) ? "shortcut active" : "16 MB round trip"}</small>
            </button>
            <div class="operation">GELU<small>0.05 ms work</small></div>
            <button class="boundary" class:local={hasLocal("mid2", 4)} data-game-affordance="action" data-action-id="fuse-boundary-b" data-current-target={guidance?.actionId === "fuse-boundary-b"} disabled={hasLocal("mid2", 4)} onclick={() => fuseChainBoundary("mid2", 4)} data-testid="boundary-b">
              <span class="parcel-mini">B</span><strong>{hasLocal("mid2", 4) ? "stays nearby" : "send away + fetch back"}</strong><small>{hasLocal("mid2", 4) ? "shortcut active" : "16 MB round trip"}</small>
            </button>
            <div class="operation">+ residual<small>0.03 ms work</small></div>
          </section>
          <p class="touch-prompt">The orange halo marks the recommended click; every dotted gate is interactive.</p>
        {:else if level.id === "layernorm"}
          {#if phase === "notice"}
            <section class="passes-scene">
              <div class="row-ribbon"><span>1,024 values</span></div>
              <div class="pass-list"><span>① find mean</span><span>② find variance</span><span>③ normalize</span></div>
              <div class="trip-count">same row parked and fetched <strong>3×</strong></div>
              <button class="big-action" data-game-affordance="action" data-action-id="layernorm-try-flow" data-current-target={guidance?.actionId === "layernorm-try-flow"} onclick={tryLayernormFlow}>Try one continuous pass →</button>
            </section>
          {:else if phase === "failure"}
            <section class="dependency-scene" data-testid="layernorm-failure">
              <div class="conveyor"><span>[2, 4]</span><span>[8, 10]</span><i>→</i></div>
              <div class="blocked-station"><strong>variance</strong><small>needs the final mean</small></div>
              <div class="changing-number">mean so far <strong>3</strong><i>becomes</i><strong>6</strong></div>
              <h2>The future changes what “distance from the mean” means.</h2>
              <p>Instead of hiding this obstruction, let’s try it on four numbers.</p>
              <button class="big-action" data-game-affordance="action" data-action-id="layernorm-open-lab" data-current-target={guidance?.actionId === "layernorm-open-lab"} onclick={openWelfordLab}>Open the 4-number experiment</button>
            </section>
          {:else if phase === "lab"}
            <section class="backpack-lab" data-testid="welford-lab">
              <div class="blocks"><span class:consumed={labStep >= 1}>[2, 4]</span><span class:consumed={labStep >= 2}>[8, 10]</span></div>
              <div class="backpack">
                <div><span>count</span><strong>{labStep === 0 ? "—" : labStep === 1 ? "2" : "4"}</strong></div>
                <div><span>mean</span><strong>{labStep === 0 ? "—" : labStep === 1 ? "3" : "6"}</strong></div>
                <div><span>spread M2</span><strong>{labStep === 0 ? "—" : labStep === 1 ? "2" : "40"}</strong></div>
              </div>
              {#if labStep === 2}<div class="moment-proof"><span>distances from mean 6</span><strong>(−4)² + (−2)² + 2² + 4² = 40</strong><small>M2 is the running sum of squared distances.</small></div>{/if}
              <p>{labStep < 2 ? "The backpack has three slots. Feed the next block." : "Four input values are gone, but these three numbers preserve exactly what variance needs."}</p>
              {#if labStep < 2}<button class="big-action" data-game-affordance="action" data-action-id="welford-feed" data-current-target={guidance?.actionId === "welford-feed"} onclick={feedMomentBlock}>Feed {labStep === 0 ? "[2, 4]" : "[8, 10]"} →</button>{:else}<button class="big-action" data-game-affordance="action" data-action-id="welford-install" data-current-target={guidance?.actionId === "welford-install"} onclick={installWelford}>Carry this backpack through LayerNorm</button>{/if}
            </section>
          {:else}
            <section class="build-scene">
              <div class="earned-tool"><span>🎒</span><div><small>NOW IT HAS A NAME</small><strong>Welford running moments</strong><p>The carried state is (count, mean, M2).</p></div></div>
              <div class="build-actions">
                <button data-game-affordance="action" data-action-id="layernorm-keep-nearby" data-current-target={guidance?.actionId === "layernorm-keep-nearby"} disabled={hasLocal("mid1", 2)} onclick={keepPassesNearby}><span>1</span><strong>Keep each chunk nearby</strong><small>remove both row round trips</small></button>
                <button data-game-affordance="action" data-action-id="layernorm-stream" data-current-target={guidance?.actionId === "layernorm-stream"} disabled={!hasLocal("mid1", 2) || hasRowStream()} onclick={streamRows}><span>2</span><strong>Flow 128 values at a time</strong><small>carry the backpack between blocks</small></button>
              </div>
            </section>
          {/if}
        {:else if level.id === "softmax"}
          {#if phase === "notice"}
            <section class="passes-scene softmax-scene">
              <div class="row-ribbon"><span>2,048 scores</span></div>
              <div class="pass-list"><span>① find max</span><span>② sum exponentials</span><span>③ normalize</span></div>
              <div class="trip-count">whole score row crossed <strong>3×</strong></div>
              <button class="big-action" data-game-affordance="action" data-action-id="softmax-try-flow" data-current-target={guidance?.actionId === "softmax-try-flow"} onclick={trySoftmaxFlow}>Try one continuous pass →</button>
            </section>
          {:else if phase === "failure"}
            <section class="dependency-scene ruler-scene" data-testid="softmax-failure">
              <div class="score-bars"><i style="height:42%">1</i><i style="height:58%">2</i><i class="future" style="height:92%">4</i></div>
              <div class="moving-ruler"><span>current max 2</span><i>→</i><strong>future max 4</strong></div>
              <h2>A later maximum moves the ruler.</h2>
              <p>Exponentials counted against max 2 cannot simply be added to values counted against max 4.</p>
              <button class="big-action" data-game-affordance="action" data-action-id="softmax-open-lab" data-current-target={guidance?.actionId === "softmax-open-lab"} onclick={openSoftmaxLab}>Try it with [2, 1] then [4]</button>
            </section>
          {:else if phase === "lab"}
            <section class="softmax-lab" data-testid="softmax-lab">
              <div class="block-reveal">
                <div class:visible={labStep >= 1}><small>BLOCK 1</small><strong>[2, 1]</strong><span>m = 2 · ℓ = 1.368</span></div>
                <div class:visible={labStep >= 2}><small>BLOCK 2</small><strong>[4]</strong><span>new m = 4</span></div>
              </div>
              {#if labStep < 2}
                <button class="big-action" data-game-affordance="action" data-action-id="softmax-reveal" data-current-target={guidance?.actionId === "softmax-reveal"} onclick={revealSoftmaxBlock}>Reveal {labStep === 0 ? "first" : "next"} block</button>
              {:else if softmaxChoice === "none"}
                <h2>What should happen to the old subtotal?</h2>
                <div class="choice-row"><button data-game-affordance="action" data-action-id="softmax-keep-wrong" onclick={chooseOldSubtotal}>Keep 1.368 and add</button><button data-game-affordance="action" data-action-id="softmax-rescale" data-current-target={guidance?.actionId === "softmax-rescale"} onclick={chooseRescale}>Shrink it to the new ruler</button></div>
              {:else if softmaxChoice === "wrong"}
                <div class="wrong-result"><strong>2.368 ✕</strong><span>The old bars were measured from max 2, not max 4.</span></div>
                <button class="big-action" data-game-affordance="action" data-action-id="softmax-rescale" data-current-target={guidance?.actionId === "softmax-rescale"} onclick={chooseRescale}>Rescale the old subtotal</button>
              {:else}
                <div class="correction-card" data-testid="softmax-correction"><span>old subtotal</span><strong>1.368 × exp(2 − 4) + 1 = 1.185</strong><p>The factor 0.135 shrinks every earlier contribution into the new maximum’s coordinate system.</p></div>
                <button class="big-action" data-game-affordance="action" data-action-id="softmax-install" data-current-target={guidance?.actionId === "softmax-install"} onclick={installOnlineSoftmax}>Carry the ruler m and subtotal ℓ</button>
              {/if}
            </section>
          {:else}
            <section class="build-scene">
              <div class="earned-tool"><span>📏</span><div><small>NOW IT HAS A NAME</small><strong>Online softmax</strong><p><b>m</b> is the moving ruler. <b>ℓ</b> is the repaired subtotal. Earlier work scales by exp(m_old − m_new).</p></div></div>
              <div class="build-actions">
                <button data-game-affordance="action" data-action-id="softmax-keep-nearby" data-current-target={guidance?.actionId === "softmax-keep-nearby"} disabled={hasLocal("mid1", 2)} onclick={keepPassesNearby}><span>1</span><strong>Keep score chunks nearby</strong><small>remove both full-row round trips</small></button>
                <button data-game-affordance="action" data-action-id="softmax-stream" data-current-target={guidance?.actionId === "softmax-stream"} disabled={!hasLocal("mid1", 2) || hasRowStream()} onclick={streamRows}><span>2</span><strong>Flow 128 scores at a time</strong><small>carry m and ℓ between blocks</small></button>
              </div>
            </section>
          {/if}
        {:else}
          <section class="attention-scene" data-testid="attention-capstone">
            <div class="attention-machine">
              <div class="attn-op">Q × Kᵀ</div>
              <div class="giant-parcel" class:removed={faDone("fuse-qk-softmax")}><strong>scores</strong><span>N × N</span><small>{faDone("fuse-qk-softmax") ? "kept in blocks" : "24 MB parked"}</small></div>
              <div class="attn-op">softmax</div>
              <div class="giant-parcel magenta" class:removed={faDone("fuse-softmax-pv")}><strong>probabilities</strong><span>N × N</span><small>{faDone("fuse-softmax-pv") ? "kept in blocks" : "24 MB parked"}</small></div>
              <div class="attn-op">× V</div>
            </div>
            <div class="earned-toolbox">
              <h2>Your four earned ideas</h2>
              <p>No new mechanic here. Apply each idea to the bottleneck it describes.</p>
              <button class:done={faDone("admit-online-softmax")} data-game-affordance="action" data-action-id="fa-carry" data-current-target={guidance?.actionId === "fa-carry"} onclick={() => performCapstone(0)} disabled={faDone("admit-online-softmax")}><span>📏</span><strong>Carry m and ℓ</strong><small>reuse the moving-ruler lesson</small></button>
              <button class:done={faDone("fuse-qk-softmax")} data-game-affordance="action" data-action-id="fa-scores" data-current-target={guidance?.actionId === "fa-scores"} onclick={() => performCapstone(1)} disabled={faDone("fuse-qk-softmax")}><span>↝</span><strong>Keep score blocks nearby</strong><small>the first giant parcel</small></button>
              <button class:done={faDone("fuse-softmax-pv")} data-game-affordance="action" data-action-id="fa-probabilities" data-current-target={guidance?.actionId === "fa-probabilities"} onclick={() => performCapstone(2)} disabled={faDone("fuse-softmax-pv")}><span>↝</span><strong>Keep probability blocks nearby</strong><small>the second giant parcel</small></button>
              <button class:done={faDone("tile-q")} data-game-affordance="action" data-action-id="fa-tile" data-current-target={guidance?.actionId === "fa-tile"} onclick={() => performCapstone(3)} disabled={faDone("tile-q")}><span>▦</span><strong>Work on 64 query rows</strong><small>fit the nearby workbench</small></button>
              <button class:done={faDone("stream-x")} class:blocked={blockedTool === "stream-x"} data-game-affordance="action" data-action-id="fa-stream" data-current-target={guidance?.actionId === "fa-stream"} onclick={() => performCapstone(4)} disabled={faDone("stream-x")}><span>≈</span><strong>Flow 32 key columns</strong><small>continue block by block</small></button>
            </div>
          </section>
        {/if}

        {#if phase === "complete"}
          <section class="completion" data-testid="level-completion">
            <div class="completion-burst">✓</div>
            <div>
              <p class="eyebrow">MACHINE RUNNING</p>
              <h2>{level.id === "attention" ? "You built the FlashAttention schedule." : "You made the row stay close and keep moving."}</h2>
              <p>{level.id === "fuse-chain" ? "Two temporary values never touch slow memory. That is fusion." : level.id === "layernorm" ? "Welford’s three-number backpack replaced three whole-row trips." : level.id === "softmax" ? "The (m, ℓ) state repairs the old subtotal whenever the ruler moves." : "The N×N score and probability arrays are no longer materialized in slow memory."}</p>
              <div class="completion-numbers"><span><b>{formatBytes(trafficBytes(level.baseline, level.id))}</b> before</span><i>→</i><span><b>{formatBytes(trafficBytes(term, level.id))}</b> now</span></div>
            </div>
            <div class="completion-actions">
              <button class="paper-button" data-game-affordance="action" data-action-id="read-notation" onclick={() => (notationOpen = !notationOpen)}>{notationOpen ? "Hide" : "Read"} the paper shorthand</button>
              <button class="next-button" data-game-affordance="action" data-action-id="lesson-map" data-current-target={guidance?.actionId === "lesson-map"} onclick={showMap}>Back to lesson map →</button>
            </div>
          </section>
          {#if notationOpen}
            <section class="earned-notation" data-testid="earned-notation">
              <header><div><small>EARNED COMPRESSION</small><h2>The same machine, written as a Neural Circuit Diagram</h2></div><p>Boxes are stations. Wires are parcels. Warm regions stayed nearby. The compact rows above count traffic (Hₗ₁) and nearby space (Mₗ₁).</p></header>
              <div class="paper-canvas">
                <NcdRenderer
                  {term}
                  paintLevel="l1"
                  onPartitionDrop={noopPartition}
                  onPartitionPreview={noopPartition}
                  onResidencyDrop={noopResidency}
                  onResidencyPreview={noopResidency}
                  onLemmaDrop={noopLemma}
                  onPreviewClear={noop}
                />
              </div>
            </section>
          {/if}
        {/if}

        {#if level.id !== "fuse-chain" || phase === "complete"}<footer class="notation-ledger">
          <span>PLAIN WORDS</span><strong>memory traffic</strong><i>becomes</i><code>Hₗ₁</code>
          <span>PLAIN WORDS</span><strong>nearby space needed</strong><i>becomes</i><code>Mₗ₁</code>
          <small>These symbols are labels for quantities you have already changed.</small>
        </footer>{/if}
      </section>
    {/if}
  </main>
{/if}

<style>
  :global(html:has(.learning-game)), :global(body:has(.learning-game)) { background: #0d1421; }
  .learning-game {
    --night: #0d1421; --night2: #162238; --paper: #fff8e8; --ink: #172033;
    --warm: #ffbd59; --orange: #ff835d; --mint: #61deb6; --blue: #72b7ff; --pink: #ef8bc7;
    position: fixed; inset: 0; overflow-y: auto; color: #f7f3e8; font-family: Inter, ui-rounded, system-ui, sans-serif;
    background: radial-gradient(circle at 10% 5%, #1a2b49, transparent 38rem), radial-gradient(circle at 88% 90%, rgba(34,132,121,.16), transparent 35rem), var(--night);
  }
  .learning-game, .learning-game * { box-sizing: border-box; }
  button { font: inherit; }
  .world-header { height: 66px; position: sticky; top: 0; z-index: 50; display: flex; align-items: center; justify-content: space-between; padding: 0 28px; background: rgba(13,20,33,.9); backdrop-filter: blur(18px); border-bottom: 1px solid rgba(255,255,255,.08); }
  .brand-button { border: 0; color: white; background: transparent; font-weight: 850; font-size: 16px; cursor: pointer; }
  .brand-button span { display: inline-grid; place-items: center; width: 29px; height: 29px; margin-right: 8px; color: var(--ink); background: var(--warm); border-radius: 50%; }
  .world-header nav { display: flex; gap: 6px; }
  .world-header nav button, .back-button, .undo-button { border: 0; padding: 9px 12px; color: #9baabe; background: transparent; border-radius: 11px; cursor: pointer; }
  .world-header nav button:hover, .world-header nav button.active, .back-button:hover, .undo-button:hover { color: white; background: rgba(255,255,255,.07); }
  .lesson-map { width: min(1080px, calc(100% - 40px)); margin: 0 auto; padding: 70px 0; }
  .map-intro { max-width: 700px; margin-bottom: 46px; }
  .eyebrow { margin: 0 0 8px; color: var(--warm); font-size: 11px; font-weight: 900; letter-spacing: .16em; }
  .map-intro h1, .mission h1 { margin: 0; font-size: clamp(38px, 6vw, 68px); line-height: .98; letter-spacing: -.055em; }
  .map-intro > p:last-child, .mission > div > p:last-child { color: #b0bdd0; font-size: 18px; line-height: 1.5; }
  .map-path { position: relative; display: grid; gap: 14px; }
  .map-path::before { content: ""; position: absolute; left: 30px; top: 35px; bottom: 35px; width: 3px; background: linear-gradient(var(--warm), var(--blue), var(--pink), var(--mint)); opacity: .45; }
  .map-card { position: relative; display: grid; grid-template-columns: 62px minmax(0,1fr) auto; align-items: center; gap: 18px; width: 100%; padding: 17px 20px 17px 0; border: 1px solid rgba(255,255,255,.09); color: white; text-align: left; background: rgba(255,255,255,.045); border-radius: 20px; cursor: pointer; transition: transform .2s, background .2s; }
  .map-card:hover { transform: translateX(7px); background: rgba(255,255,255,.08); }
  .map-index { z-index: 1; display: grid; place-items: center; width: 44px; height: 44px; margin-left: 9px; color: var(--ink); background: var(--paper); border: 5px solid var(--night2); border-radius: 50%; font-weight: 900; }
  .map-card.complete .map-index { background: var(--mint); }
  .map-copy { display: grid; gap: 3px; }
  .map-copy small { color: var(--warm); font-size: 10px; font-weight: 900; letter-spacing: .13em; }
  .map-copy strong { font-size: 22px; letter-spacing: -.025em; }
  .map-copy em { color: #8fa0b7; font-size: 13px; font-style: normal; }
  .map-arrow { color: var(--warm); font-size: 26px; }
  .map-result { color: var(--mint); font-size: 12px; font-weight: 800; }
  .map-note { margin-top: 22px; display: flex; align-items: center; gap: 16px; padding: 16px 19px; color: #9daabd; background: rgba(255,189,89,.07); border-radius: 16px; }
  .map-note strong { color: var(--warm); }
  .map-note button { margin-left: auto; border: 0; color: white; background: transparent; text-decoration: underline; cursor: pointer; }
  .lesson { width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 20px 0 90px; }
  .lesson-topline { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; color: #77869d; font-size: 12px; }
  .back-button { justify-self: start; }
  .undo-button { justify-self: end; }
  .undo-button:disabled { opacity: .35; cursor: default; }
  .mission { display: grid; grid-template-columns: minmax(0,1fr) 270px; gap: 50px; align-items: end; padding: 45px 0 28px; }
  .physical-score { padding: 19px; background: rgba(255,255,255,.055); border: 1px solid rgba(255,255,255,.1); border-radius: 20px; }
  .physical-score > div { display: flex; justify-content: space-between; align-items: baseline; padding: 5px 0; }
  .physical-score span { color: #93a2b8; font-size: 12px; }
  .physical-score strong { font-size: 22px; font-variant-numeric: tabular-nums; }
  .physical-score small { display: block; margin-top: 9px; padding-top: 9px; color: var(--warm); border-top: 1px solid rgba(255,255,255,.1); }
  .physical-score.won { background: rgba(97,222,182,.12); }
  .physical-score.won strong, .physical-score.won small { color: var(--mint); }
  .physical-score .space-row { margin-top: 5px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,.08); }
  .physical-score .space-row strong { color: #b9e9db; font-size: 15px; }
  .action-guide { position: relative; display: grid; grid-template-columns: auto minmax(0,1fr) auto; gap: 16px; align-items: center; margin-bottom: 14px; padding: 17px 20px; color: var(--ink); background: linear-gradient(110deg,#fff2bd,#ffcf72); border: 2px solid #ffb33b; border-radius: 18px; box-shadow: 0 12px 32px rgba(255,179,59,.13); }
  .guide-kicker { padding: 6px 9px; color: #5d3900; background: rgba(255,255,255,.55); border-radius: 8px; font-size: 9px; font-weight: 950; letter-spacing: .14em; }
  .action-guide strong { display: block; font-size: 17px; line-height: 1.3; }
  .action-guide p { margin: 3px 0 0; color: #70521e; font-size: 12px; }
  .guide-arrow { font-size: 28px; animation: guide-nudge 1.8s ease-in-out infinite; }
  .feedback { min-height: 64px; display: flex; align-items: center; gap: 14px; margin-bottom: 18px; padding: 14px 18px; color: #cad6e5; background: rgba(114,183,255,.08); border: 1px solid rgba(114,183,255,.16); border-radius: 16px; }
  .feedback span { color: var(--blue); font-size: 23px; }
  .feedback p { margin: 0; line-height: 1.45; }
  .chain-machine { min-height: 330px; display: grid; grid-template-columns: 1fr 1.15fr 1fr 1.15fr 1fr; gap: 14px; align-items: center; padding: 44px; background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.08); border-radius: 28px; }
  .operation, .attn-op { display: grid; place-items: center; min-height: 110px; padding: 16px; color: var(--ink); background: var(--paper); border-radius: 20px; font-size: 22px; font-weight: 850; box-shadow: 0 15px 35px rgba(0,0,0,.22); }
  .operation small { color: #778091; font-size: 10px; font-weight: 600; }
  .boundary { position: relative; min-height: 170px; display: grid; place-items: center; align-content: center; gap: 4px; color: #eaf5ff; background: linear-gradient(180deg,rgba(83,142,206,.2),rgba(83,142,206,.1)); border: 2px solid rgba(114,183,255,.68); border-radius: 24px; cursor: pointer; box-shadow: inset 0 0 0 1px rgba(255,255,255,.05); }
  .boundary::before { content: "CLICK"; position: absolute; top: 10px; right: 11px; padding: 3px 6px; color: #bfe0ff; background: rgba(8,28,51,.65); border-radius: 6px; font-size: 8px; font-weight: 950; letter-spacing: .12em; }
  .boundary:hover { background: rgba(83,142,206,.3); border-color: #a4d4ff; transform: translateY(-3px); }
  .boundary.local { color: #3d2b0a; background: rgba(255,189,89,.92); border-color: var(--warm); cursor: default; animation: pop .45s both; }
  .boundary.local::before { content: "DONE"; color: #5c3700; background: rgba(255,255,255,.48); }
  .boundary small { opacity: .75; }
  .parcel-mini { display: grid; place-items: center; width: 44px; height: 38px; color: var(--ink); background: var(--warm); border-radius: 11px; font-size: 20px; font-weight: 900; }
  .boundary.local .parcel-mini { background: #fff3bc; }
  .touch-prompt { color: #8392a8; text-align: center; }
  .passes-scene, .dependency-scene, .backpack-lab, .softmax-lab, .build-scene { min-height: 390px; padding: 42px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); border-radius: 28px; }
  .passes-scene { min-height: 310px; padding: 28px 42px; }
  .row-ribbon { height: 76px; display: grid; place-items: center; color: var(--ink); background: repeating-linear-gradient(90deg,#ffda79 0 28px,#f7c75e 28px 56px); border-radius: 18px; font-weight: 900; }
  .pass-list { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 22px 0; }
  .pass-list span { padding: 14px; color: #c9d5e5; background: rgba(255,255,255,.06); border-radius: 13px; text-align: center; }
  .trip-count { margin: 10px 0 26px; color: #98a8bc; text-align: center; }
  .trip-count strong { color: var(--blue); font-size: 25px; }
  .big-action { display: block; margin: 0 auto; padding: 14px 20px; border: 0; color: var(--ink); background: var(--warm); border-radius: 14px; font-weight: 900; cursor: pointer; }
  .big-action:hover { transform: translateY(-2px); filter: brightness(1.05); }
  .dependency-scene { text-align: center; }
  .conveyor { display: flex; justify-content: center; gap: 16px; align-items: center; }
  .conveyor span { padding: 15px 20px; color: var(--ink); background: var(--paper); border-radius: 13px; font-weight: 800; }
  .blocked-station { width: 180px; margin: 20px auto 10px; padding: 18px; color: #ffd4cd; background: #6b3038; border-radius: 16px; animation: shake .35s 2; }
  .blocked-station small, .blocked-station strong { display: block; }
  .changing-number { display: flex; justify-content: center; gap: 12px; align-items: center; color: #aab7c9; }
  .changing-number strong { color: var(--warm); font-size: 26px; }
  .dependency-scene h2 { margin: 24px 0 5px; font-size: 25px; }
  .dependency-scene p { color: #9eacc0; }
  .blocks, .block-reveal { display: flex; justify-content: center; gap: 18px; margin-bottom: 28px; }
  .blocks span { padding: 18px 28px; color: var(--ink); background: var(--paper); border-radius: 15px; font-size: 20px; font-weight: 850; transition: opacity .3s, transform .3s; }
  .blocks span.consumed { opacity: .35; transform: translateY(80px) scale(.8); }
  .backpack { width: min(600px,100%); display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 0 auto 22px; padding: 17px; color: var(--ink); background: linear-gradient(145deg,#ffe9a5,var(--warm)); border-radius: 30px 30px 16px 16px; box-shadow: 0 18px 45px rgba(255,189,89,.2); }
  .backpack div { padding: 13px; text-align: center; background: rgba(255,255,255,.5); border-radius: 12px; }
  .backpack span, .backpack strong { display: block; }
  .backpack span { font-size: 11px; }
  .backpack strong { font-size: 25px; }
  .backpack-lab > p, .softmax-lab > h2 { text-align: center; color: #c4d0df; }
  .moment-proof { width: min(600px,100%); margin: -10px auto 16px; padding: 12px 16px; color: #c8d6e5; text-align: center; background: rgba(255,255,255,.06); border-radius: 12px; }
  .moment-proof span, .moment-proof strong, .moment-proof small { display: block; }
  .moment-proof strong { margin: 3px 0; color: var(--warm); font-size: 18px; }
  .moment-proof small { color: #8e9db1; }
  .score-bars { height: 150px; display: flex; justify-content: center; align-items: end; gap: 13px; }
  .score-bars i { width: 65px; display: grid; place-items: start center; padding-top: 10px; color: var(--ink); background: var(--warm); border-radius: 10px 10px 0 0; font-style: normal; font-weight: 900; }
  .score-bars .future { background: var(--pink); }
  .moving-ruler { display: flex; justify-content: center; gap: 12px; padding: 12px; color: #a8b7ca; border-top: 2px solid #8090a5; }
  .moving-ruler strong { color: var(--pink); }
  .block-reveal > div { min-width: 230px; padding: 21px; opacity: .16; transform: scale(.93); color: var(--ink); background: var(--paper); border-radius: 17px; transition: .35s; }
  .block-reveal > div.visible { opacity: 1; transform: none; }
  .block-reveal small, .block-reveal strong, .block-reveal span { display: block; }
  .block-reveal strong { margin: 5px 0; font-size: 24px; }
  .choice-row { display: flex; justify-content: center; gap: 12px; }
  .choice-row button { padding: 15px 18px; border: 1px solid rgba(255,255,255,.15); color: white; background: rgba(255,255,255,.07); border-radius: 14px; cursor: pointer; }
  .choice-row button:hover { border-color: var(--warm); }
  .wrong-result, .correction-card { width: min(680px,100%); margin: 18px auto; padding: 18px; text-align: center; border-radius: 16px; }
  .wrong-result { color: #ffd4cc; background: rgba(180,61,66,.25); }
  .wrong-result strong, .wrong-result span, .correction-card span, .correction-card strong { display: block; }
  .wrong-result strong { font-size: 27px; }
  .correction-card { color: var(--ink); background: #d8f7e9; }
  .correction-card strong { margin: 7px; font-size: 23px; }
  .earned-tool { display: flex; align-items: center; gap: 20px; padding: 21px; color: var(--ink); background: linear-gradient(135deg,#ffe59d,#ffbf64); border-radius: 20px; }
  .earned-tool > span { font-size: 42px; }
  .earned-tool small, .earned-tool strong { display: block; }
  .earned-tool small { font-size: 9px; font-weight: 900; letter-spacing: .14em; }
  .earned-tool strong { font-size: 23px; }
  .earned-tool p { margin: 4px 0 0; }
  .build-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 22px; }
  .build-actions button { min-height: 115px; display: grid; grid-template-columns: auto 1fr; column-gap: 12px; align-content: center; padding: 17px; color: white; text-align: left; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 17px; cursor: pointer; }
  .build-actions button > span { grid-row: 1 / 3; display: grid; place-items: center; width: 34px; height: 34px; color: var(--ink); background: var(--mint); border-radius: 50%; font-weight: 900; }
  .build-actions button strong, .build-actions button small { display: block; }
  .build-actions button small { color: #91a0b5; }
  .build-actions button:disabled { opacity: .35; cursor: default; }
  .attention-scene { display: grid; grid-template-columns: minmax(0,1.35fr) minmax(310px,.65fr); gap: 18px; }
  .attention-machine, .earned-toolbox { padding: 26px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); border-radius: 24px; }
  .attention-machine { display: grid; grid-template-columns: 1fr .8fr 1fr .8fr 1fr; align-items: center; gap: 10px; }
  .attn-op { min-height: 90px; font-size: 17px; }
  .giant-parcel { min-height: 210px; display: grid; place-items: center; align-content: center; color: #dcecff; background: rgba(81,144,217,.22); border: 2px dashed rgba(114,183,255,.45); border-radius: 18px; transition: .5s; }
  .giant-parcel.magenta { color: #ffe1f3; background: rgba(209,92,164,.2); border-color: rgba(239,139,199,.45); }
  .giant-parcel span { font-size: 25px; font-weight: 900; }
  .giant-parcel small { margin-top: 8px; }
  .giant-parcel.removed { min-height: 85px; color: #3b2b0b; background: var(--warm); border-color: var(--warm); }
  .giant-parcel.removed span { font-size: 13px; }
  .earned-toolbox h2 { margin: 0; }
  .earned-toolbox > p { color: #91a0b6; }
  .earned-toolbox > button { width: 100%; display: grid; grid-template-columns: 35px 1fr; margin-top: 7px; padding: 12px; color: white; text-align: left; background: rgba(255,255,255,.055); border: 1px solid rgba(255,255,255,.13); border-radius: 12px; cursor: pointer; }
  .earned-toolbox > button span { grid-row: 1 / 3; font-size: 20px; }
  .earned-toolbox > button strong, .earned-toolbox > button small { display: block; }
  .earned-toolbox > button small { color: #8796ac; }
  .earned-toolbox > button:hover { background: rgba(255,255,255,.12); border-color: rgba(255,189,89,.7); transform: translateX(-3px); }
  .earned-toolbox > button.done { opacity: .4; }
  .earned-toolbox > button.blocked { color: #ffc9c0; background: rgba(180,61,66,.2); animation: shake .35s 2; }
  .completion { margin-top: 18px; display: grid; grid-template-columns: auto 1fr auto; gap: 22px; align-items: center; padding: 25px; color: var(--ink); background: #dcf9ee; border-radius: 24px; animation: rise .5s both; }
  .completion-burst { display: grid; place-items: center; width: 62px; height: 62px; color: #073d31; background: var(--mint); border-radius: 50%; font-size: 30px; font-weight: 900; }
  .completion h2 { margin: 0 0 6px; font-size: 25px; }
  .completion p { margin: 0; color: #4d665f; }
  .completion-numbers { display: flex; gap: 10px; align-items: center; margin-top: 13px; }
  .completion-numbers span { padding: 7px 10px; background: rgba(255,255,255,.56); border-radius: 9px; }
  .completion-actions { display: grid; gap: 8px; }
  .completion-actions button { padding: 12px 14px; border-radius: 12px; font-weight: 800; cursor: pointer; }
  .paper-button { color: #21493f; background: transparent; border: 1px solid #89b7aa; }
  .next-button { color: white; background: #163a32; border: 0; }
  .earned-notation { margin-top: 18px; color: var(--ink); background: #f7f0e2; border-radius: 24px; overflow: hidden; }
  .earned-notation > header { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: center; padding: 22px 28px; }
  .earned-notation small { color: #9a632b; font-weight: 900; letter-spacing: .14em; }
  .earned-notation h2 { margin: 4px 0 0; font-family: Georgia,serif; font-weight: 500; }
  .earned-notation p { color: #5e655f; line-height: 1.45; }
  .paper-canvas { height: 550px; display: flex; }
  .paper-canvas :global(.ncd-viewport) { flex: 1; }
  .notation-ledger { margin-top: 22px; display: grid; grid-template-columns: auto auto auto auto; gap: 7px 13px; align-items: baseline; padding: 18px; color: #94a2b7; border-top: 1px solid rgba(255,255,255,.08); }
  .notation-ledger > span { color: #6f7f95; font-size: 9px; font-weight: 900; letter-spacing: .13em; }
  .notation-ledger > strong { color: #cad4e2; }
  .notation-ledger > i { font-size: 11px; }
  .notation-ledger code { color: var(--warm); font-family: Georgia,serif; font-size: 18px; }
  .notation-ledger small { grid-column: 1 / -1; }
  .sandbox-shell { position: fixed; inset: 0; z-index: 100; display: flex; flex-direction: column; color: var(--foreground); background: var(--background); }
  .sandbox-back { height: 36px; flex: none; border: 0; border-bottom: 1px solid var(--border); color: var(--foreground); background: var(--panel); cursor: pointer; }
  [data-game-affordance]:not(:disabled) { cursor: pointer; }
  [data-game-affordance="action"]:not(:disabled) { transition: transform .18s, box-shadow .18s, border-color .18s, background .18s; }
  [data-current-target="true"]:not(:disabled) { position: relative; z-index: 4; outline: 3px solid var(--warm); outline-offset: 4px; box-shadow: 0 0 0 8px rgba(255,189,89,.12), 0 0 30px rgba(255,189,89,.32); animation: target-breathe 1.9s ease-in-out infinite; }
  @keyframes pop { from { transform: scale(.85); opacity: .4; } }
  @keyframes shake { 25% { transform: translateX(-5px); } 75% { transform: translateX(5px); } }
  @keyframes rise { from { opacity: 0; transform: translateY(12px); } }
  @keyframes target-breathe { 50% { outline-color: #ffe199; box-shadow: 0 0 0 12px rgba(255,189,89,.06), 0 0 38px rgba(255,189,89,.42); } }
  @keyframes guide-nudge { 50% { transform: translate(4px,4px); } }
  @media (max-width: 850px) {
    .world-header nav button:last-child { display: none; }
    .mission { grid-template-columns: 1fr; gap: 12px; }
    .chain-machine { grid-template-columns: 1fr; }
    .boundary { min-height: 90px; }
    .attention-scene { grid-template-columns: 1fr; }
    .attention-machine { grid-template-columns: 1fr; }
    .giant-parcel { min-height: 100px; }
    .completion { grid-template-columns: auto 1fr; }
    .completion-actions { grid-column: 1 / -1; }
    .map-note { align-items: flex-start; flex-direction: column; }
    .map-note button { margin-left: 0; }
  }
</style>
