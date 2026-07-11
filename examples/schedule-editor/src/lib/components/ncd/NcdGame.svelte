<script lang="ts">
  import { applyMove, cloneTerm, napkinCost } from "../../ncd/model";
  import { GAME_LEVELS } from "../../ncd/game-levels";
  import type { NcdTerm } from "../../ncd/types";

  let { onExit }: { onExit: () => void } = $props();

  const ROWS_PER_STEP = 2048;
  const base = GAME_LEVELS[0].baseline;
  let current = $state<NcdTerm>(cloneTerm(base));
  let dragging = $state(false);
  let dragX = $state(0);
  let dragY = $state(0);
  let solved = $state(false);
  let showNotation = $state(false);
  let miss = $state(false);

  const trafficMb = $derived(
    Math.round(
      (napkinCost(current).transferBytesByLevel.l1 * ROWS_PER_STEP) /
        (1024 * 1024),
    ),
  );

  function beginDrag(event: PointerEvent): void {
    if (solved) return;
    event.preventDefault();
    dragging = true;
    miss = false;
    dragX = event.clientX;
    dragY = event.clientY;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  }

  function moveDrag(event: PointerEvent): void {
    if (!dragging) return;
    event.preventDefault();
    dragX = event.clientX;
    dragY = event.clientY;
  }

  function endDrag(event: PointerEvent): void {
    if (!dragging) return;
    event.preventDefault();
    dragging = false;
    const target = document.elementFromPoint(event.clientX, event.clientY);
    if (target?.closest('[data-nearby-drop="true"]')) {
      solve();
    } else {
      miss = true;
      window.setTimeout(() => (miss = false), 1200);
    }
  }

  function solve(): void {
    if (solved) return;
    current = applyMove(current, {
      op: "recolor",
      wireId: "mid1",
      column: 2,
      before: "l0",
      after: "l1",
    });
    solved = true;
    dragging = false;
  }

  function reset(): void {
    current = cloneTerm(base);
    solved = false;
    showNotation = false;
    dragging = false;
    miss = false;
  }

  function cancel(): void {
    dragging = false;
    miss = false;
  }
</script>

<svelte:window onpointermove={moveDrag} onpointerup={endDrag} onkeydown={(event) => event.key === "Escape" && cancel()} />

<main class="memory-game" data-testid="memory-game">
  <header class="game-header">
    <div class="brand"><span class="brand-mark">↝</span> Memory Garden</div>
    <div class="step-markers" aria-label="Lesson progress">
      <span class="active"></span><span></span><span></span><span></span><span></span>
    </div>
    <button class="quiet-button" onclick={onExit}>Schedule editor</button>
  </header>

  <section class="lesson-stage" class:solved aria-live="polite">
    <div class="lesson-heading">
      <p class="kicker">A 10-second experiment</p>
      <h1>{solved ? "That was the shortcut." : "Can you shorten this trip?"}</h1>
      <p class="instruction">
        {#if solved}
          Watch where the bright parcel travels now.
        {:else}
          Drag the glowing parcel onto <strong>KEEP NEARBY</strong>.
        {/if}
      </p>
    </div>

    <div class="traffic-card" class:changed={solved}>
      <span>Memory traffic</span>
      <strong data-testid="traffic-value">{trafficMb} MB</strong>
      <small>one training step · {ROWS_PER_STEP.toLocaleString()} rows</small>
    </div>

    <div class="machine" class:miss aria-label="Bias and GELU computation with a slow-memory detour">
      <div class="station bias-station">
        <span class="station-number">01</span>
        <strong>Add bias</strong>
        <small>quick arithmetic</small>
      </div>

      <div class="nearby-zone" data-nearby-drop="true" onclick={solve} role="button" tabindex="0" onkeydown={(event) => (event.key === "Enter" || event.key === " ") && solve()}>
        <span>KEEP NEARBY</span>
        <div class="nearby-shelf"></div>
      </div>

      <div class="station gelu-station">
        <span class="station-number">02</span>
        <strong>GELU</strong>
        <small>quick arithmetic</small>
      </div>

      <svg class="routes" viewBox="0 0 1000 420" aria-hidden="true" preserveAspectRatio="none">
        <path class="direct-route" d="M 235 120 C 380 80, 610 80, 765 120" />
        <path class="detour-route" class:retired={solved} d="M 235 145 C 275 310, 400 340, 500 340 C 615 340, 720 305, 765 145" />
        <path class="active-route" class:shortcut={solved} d={solved ? "M 235 120 C 390 80, 610 80, 765 120" : "M 235 145 C 275 310, 400 340, 500 340 C 615 340, 720 305, 765 145"} />
      </svg>

      <div class="slow-memory" class:retired={solved}>
        <span class="depot-light"></span>
        <strong>SLOW MEMORY</strong>
        <small>far away · lots of room</small>
      </div>

      {#if !solved}
        <button
          class="parcel"
          class:dragging
          aria-label="Temporary result parcel. Drag to Keep Nearby."
          onpointerdown={beginDrag}
          style={dragging ? `position:fixed;left:${dragX - 34}px;top:${dragY - 28}px` : ""}
        >
          <span>A</span><small>temporary</small>
        </button>
      {:else}
        <div class="parcel parked"><span>A</span><small>temporary</small></div>
      {/if}

      {#if miss}
        <div class="miss-note">That still sends it away. Try the warm shelf.</div>
      {/if}
    </div>

    {#if solved}
      <div class="interpretation" data-testid="interpretation">
        <div class="saved-badge">−16 MB</div>
        <div>
          <h2>You removed one round trip.</h2>
          <p>The temporary result stays close, so it is not written to slow memory and immediately read back.</p>
        </div>
        <button class="reveal-button" onclick={() => (showNotation = !showNotation)}>
          {showNotation ? "Hide the shorthand" : "Show me the shorthand"}
        </button>
      </div>
    {/if}

    {#if showNotation}
      <div class="notation-reveal" data-testid="notation-reveal">
        <div class="notation-copy">
          <p class="kicker">The mark you just earned</p>
          <h2>Warm color means “keep this value nearby.”</h2>
          <p>The paper diagram compresses the whole shortcut into one colored segment. It is a record of your decision, not a new rule.</p>
        </div>
        <div class="tiny-ncd" aria-label="NCD shorthand for the nearby intermediate">
          <div class="axis-stack"><i>A</i><i class="axis">r̄ = 1024</i></div>
          <div class="warm-segment"></div>
          <div class="function-box">GELU</div>
          <span class="notation-caption">nearby memory</span>
        </div>
        <div class="reveal-actions">
          <button class="primary-button" disabled title="The full lesson arrives in Commit B">Continue to the chain <span>→</span></button>
          <button class="text-button" onclick={reset}>Run it again</button>
        </div>
      </div>
    {/if}
  </section>
</main>

<style>
  :global(html:has(.memory-game)), :global(body:has(.memory-game)) { background: #101724; }
  .memory-game {
    --night: #101724;
    --night-2: #18243a;
    --cream: #fff8e8;
    --ink: #172033;
    --muted: #7e8aa0;
    --warm: #ffbd59;
    --warm-2: #ff8b5c;
    --cool: #78b9ff;
    --mint: #67e0bb;
    position: fixed;
    inset: 0;
    overflow: auto;
    color: var(--cream);
    font-family: Inter, ui-rounded, system-ui, sans-serif;
    background:
      radial-gradient(circle at 18% 20%, rgba(74, 99, 157, .28), transparent 34rem),
      radial-gradient(circle at 83% 80%, rgba(47, 137, 132, .14), transparent 30rem),
      var(--night);
  }
  .memory-game, .memory-game * { box-sizing: border-box; }
  .game-header { height: 64px; display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; padding: 0 28px; border-bottom: 1px solid rgba(255,255,255,.08); }
  .brand { font-weight: 780; letter-spacing: -.02em; font-size: 17px; }
  .brand-mark { display: inline-grid; place-items: center; width: 29px; height: 29px; margin-right: 8px; color: var(--night); background: var(--warm); border-radius: 50%; }
  .step-markers { display: flex; gap: 7px; }
  .step-markers span { width: 24px; height: 4px; border-radius: 9px; background: rgba(255,255,255,.13); }
  .step-markers .active { background: var(--warm); }
  button { font: inherit; }
  .quiet-button { justify-self: end; border: 0; color: #aebbd0; background: transparent; cursor: pointer; padding: 10px 12px; border-radius: 12px; }
  .quiet-button:hover { color: white; background: rgba(255,255,255,.07); }
  .lesson-stage { width: min(1120px, calc(100% - 40px)); margin: 0 auto; padding: 58px 0 80px; position: relative; }
  .lesson-heading { max-width: 670px; }
  .kicker { margin: 0 0 10px; color: var(--warm); text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 800; }
  h1 { margin: 0; font-size: clamp(34px, 5vw, 64px); line-height: .98; letter-spacing: -.055em; }
  .instruction { margin: 20px 0 0; color: #c6d0df; font-size: 18px; }
  .instruction strong { color: var(--cream); border-bottom: 2px solid var(--warm); }
  .traffic-card { position: absolute; right: 0; top: 58px; min-width: 210px; padding: 18px 20px; color: #c8d4e5; background: rgba(255,255,255,.055); border: 1px solid rgba(255,255,255,.09); border-radius: 18px; transition: background .4s, transform .4s; }
  .traffic-card span, .traffic-card small { display: block; }
  .traffic-card strong { display: block; margin: 5px 0 1px; color: white; font-size: 36px; letter-spacing: -.04em; font-variant-numeric: tabular-nums; }
  .traffic-card small { color: #8897ac; font-size: 11px; }
  .traffic-card.changed { background: rgba(103,224,187,.12); transform: scale(1.03); }
  .traffic-card.changed strong { color: var(--mint); }
  .machine { position: relative; height: 430px; margin-top: 42px; border-radius: 28px; background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018)); border: 1px solid rgba(255,255,255,.08); overflow: hidden; }
  .station { position: absolute; top: 65px; z-index: 4; width: 190px; min-height: 116px; padding: 22px; color: var(--ink); background: var(--cream); border-radius: 22px; box-shadow: 0 16px 45px rgba(0,0,0,.3); }
  .bias-station { left: 45px; }
  .gelu-station { right: 45px; }
  .station-number { position: absolute; right: 15px; top: 12px; color: #abb1ba; font-size: 11px; font-weight: 800; }
  .station strong { display: block; font-size: 22px; letter-spacing: -.03em; }
  .station small { color: #70798a; }
  .nearby-zone { position: absolute; z-index: 3; top: 58px; left: 50%; width: 260px; height: 130px; transform: translateX(-50%); display: grid; place-items: start center; padding-top: 15px; color: #ffd98b; font-size: 11px; font-weight: 900; letter-spacing: .13em; border: 2px dashed rgba(255,189,89,.5); border-radius: 24px; background: rgba(255,189,89,.06); cursor: copy; animation: breathe 2.2s ease-in-out infinite; }
  .nearby-zone:hover { background: rgba(255,189,89,.13); border-color: var(--warm); }
  .nearby-shelf { position: absolute; left: 28px; right: 28px; bottom: 23px; height: 12px; background: linear-gradient(90deg, var(--warm), var(--warm-2)); border-radius: 8px; box-shadow: 0 8px 26px rgba(255,139,92,.25); }
  .routes { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; pointer-events: none; }
  .routes path { fill: none; vector-effect: non-scaling-stroke; }
  .direct-route { stroke: rgba(255,255,255,.06); stroke-width: 16; }
  .detour-route { stroke: rgba(120,185,255,.2); stroke-width: 20; transition: opacity .5s; }
  .detour-route.retired { opacity: .12; stroke-dasharray: 4 10; }
  .active-route { stroke: var(--cool); stroke-width: 5; stroke-linecap: round; stroke-dasharray: 7 14; animation: route-flow 1s linear infinite; transition: d .5s, stroke .4s; }
  .active-route.shortcut { stroke: var(--warm); }
  .slow-memory { position: absolute; z-index: 3; bottom: 30px; left: 50%; width: 235px; height: 88px; transform: translateX(-50%); display: flex; flex-direction: column; justify-content: center; align-items: center; color: #d4e8ff; background: rgba(86,145,213,.13); border: 1px solid rgba(120,185,255,.3); border-radius: 18px; transition: opacity .5s, filter .5s; }
  .slow-memory.retired { opacity: .35; filter: grayscale(.8); }
  .slow-memory strong { font-size: 13px; letter-spacing: .12em; }
  .slow-memory small { color: #87a4c5; }
  .depot-light { position: absolute; left: 18px; top: 18px; width: 8px; height: 8px; border-radius: 50%; background: var(--cool); box-shadow: 0 0 16px var(--cool); }
  .parcel { position: absolute; z-index: 20; left: calc(50% - 35px); bottom: 42px; width: 70px; height: 58px; display: grid; place-items: center; border: 0; color: var(--ink); background: linear-gradient(145deg, #fff1ab, var(--warm)); border-radius: 14px; box-shadow: 0 10px 32px rgba(255,189,89,.35); cursor: grab; touch-action: none; user-select: none; animation: parcel-pulse 1.8s ease-in-out infinite; }
  .parcel span { font-size: 22px; font-weight: 900; line-height: .8; }
  .parcel small { font-size: 9px; font-weight: 700; }
  .parcel.dragging { margin: 0; pointer-events: none; cursor: grabbing; animation: none; transform: scale(1.08) rotate(-3deg); }
  .parcel.parked { z-index: 6; top: 105px; bottom: auto; background: linear-gradient(145deg, #ffe9a0, var(--warm-2)); animation: settle .5s both; }
  .miss-note { position: absolute; z-index: 30; left: 50%; bottom: 128px; transform: translateX(-50%); color: #ffd7d0; background: #6b3035; padding: 10px 14px; border-radius: 12px; font-weight: 700; }
  .machine.miss .slow-memory { animation: shake .3s linear 2; }
  .interpretation { margin-top: 18px; display: grid; grid-template-columns: auto 1fr auto; gap: 18px; align-items: center; padding: 20px; color: #dce7f3; background: rgba(103,224,187,.09); border: 1px solid rgba(103,224,187,.25); border-radius: 20px; animation: rise .45s both; }
  .interpretation h2 { margin: 0 0 4px; color: white; font-size: 21px; }
  .interpretation p { margin: 0; color: #afc3cf; }
  .saved-badge { min-width: 86px; padding: 13px 14px; color: #092e27; background: var(--mint); border-radius: 14px; font-size: 20px; font-weight: 900; text-align: center; }
  .reveal-button, .primary-button { border: 0; padding: 13px 17px; color: var(--ink); background: var(--warm); border-radius: 13px; font-weight: 850; cursor: pointer; }
  .reveal-button:hover, .primary-button:hover { transform: translateY(-1px); filter: brightness(1.06); }
  .notation-reveal { margin-top: 18px; display: grid; grid-template-columns: 1fr 330px; gap: 25px; padding: 30px; color: var(--ink); background: #fffaf0; border-radius: 24px; animation: rise .45s both; }
  .notation-copy h2 { margin: 0 0 9px; font-family: Georgia, serif; font-size: 28px; font-weight: 500; letter-spacing: -.025em; }
  .notation-copy p:last-child { margin: 0; color: #586173; max-width: 580px; line-height: 1.55; }
  .tiny-ncd { position: relative; min-height: 145px; display: flex; align-items: center; justify-content: center; gap: 35px; font-family: Georgia, serif; border-radius: 18px; background: #f5eedf; overflow: hidden; }
  .tiny-ncd::before { content: ""; position: absolute; top: 30px; bottom: 30px; left: 68px; right: 68px; background: rgba(255,189,89,.33); border-radius: 50%; filter: blur(1px); }
  .axis-stack { z-index: 1; display: grid; padding: 9px 12px; border-left: 2px solid #273149; }
  .axis-stack i { font-size: 20px; }
  .axis-stack .axis { font-size: 14px; border-top: 1px solid #6f746d; }
  .warm-segment { z-index: 1; width: 54px; height: 3px; background: var(--warm-2); }
  .function-box { z-index: 1; padding: 13px 10px; border: 1.5px solid #273149; background: #fffaf0; font-style: italic; }
  .notation-caption { position: absolute; bottom: 12px; color: #9a632b; font-family: Inter, sans-serif; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; }
  .reveal-actions { grid-column: 1 / -1; display: flex; align-items: center; gap: 16px; border-top: 1px solid #e5ddce; padding-top: 20px; }
  .primary-button:disabled { opacity: .45; cursor: default; }
  .text-button { border: 0; color: #5b6472; background: transparent; cursor: pointer; text-decoration: underline; text-underline-offset: 4px; }
  @keyframes route-flow { to { stroke-dashoffset: -21; } }
  @keyframes parcel-pulse { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); box-shadow: 0 17px 42px rgba(255,189,89,.5); } }
  @keyframes breathe { 50% { border-color: rgba(255,189,89,.85); background: rgba(255,189,89,.1); } }
  @keyframes shake { 25% { transform: translateX(calc(-50% - 5px)); } 75% { transform: translateX(calc(-50% + 5px)); } }
  @keyframes rise { from { opacity: 0; transform: translateY(10px); } }
  @keyframes settle { from { transform: translateY(180px) scale(.85); } }
  @media (max-width: 760px) {
    .game-header { grid-template-columns: 1fr auto; padding: 0 16px; }
    .step-markers { display: none; }
    .lesson-stage { padding-top: 32px; }
    .traffic-card { position: static; display: inline-block; margin-top: 22px; }
    .machine { height: 490px; }
    .station { width: 145px; padding: 17px; }
    .bias-station { left: 18px; }
    .gelu-station { right: 18px; }
    .nearby-zone { top: 205px; width: 220px; }
    .slow-memory { bottom: 25px; }
    .interpretation { grid-template-columns: auto 1fr; }
    .reveal-button { grid-column: 1 / -1; }
    .notation-reveal { grid-template-columns: 1fr; }
  }
</style>
