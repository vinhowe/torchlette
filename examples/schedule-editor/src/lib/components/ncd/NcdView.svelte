<script lang="ts">
import {
  CircleHelp,
  Play,
  Redo2,
  RotateCcw,
  StepForward,
  Undo2,
} from "@lucide/svelte";
import { onMount } from "svelte";
import { applyFaStep, FA_STEPS } from "../../ncd/fa-script";
import {
  applyMove,
  axisById,
  cloneTerm,
  deriveProjection,
  fromDiagram,
  inverseMove,
  napkinCost,
  onlineSoftmaxLemma,
  partitionLegality,
  recolorLegality,
  termHash,
  toDiagram,
} from "../../ncd/model";
import type {
  SurfaceEquivalence,
  SurfaceGesture,
  SurfaceJam,
} from "../../ncd/surface-layout";
import type {
  NapkinCost,
  NcdHistoryEntry,
  NcdLevel,
  NcdMove,
  NcdTerm,
  PartitionDecoration,
  PartitionKind,
} from "../../ncd/types";
import NcdRenderer from "./NcdRenderer.svelte";

let attentionBase = $state<NcdTerm | null>(null);
let matmulBase = $state<NcdTerm | null>(null);
let current = $state<NcdTerm | null>(null);
let fixture = $state<"attention" | "matmul">("attention");
let history = $state<NcdHistoryEntry[]>([]);
let redoStack = $state<NcdHistoryEntry[]>([]);
let refusal = $state("Ready for a relabeling gesture.");
let groupSize = $state(64);
let streamSize = $state(32);
let faStep = $state(0);
let loadError = $state("");
let paintLevel = $state<NcdLevel>("l1");
let previewTerm = $state<NcdTerm | null>(null);
let jam = $state<SurfaceJam | null>(null);
let equivalence = $state<SurfaceEquivalence | null>(null);
let equivalenceNonce = 0;
let equivalenceTimer: ReturnType<typeof setTimeout> | undefined;
let replayRunning = $state(false);
let activeGesture = $state<SurfaceGesture | null>(null);
let helpOpen = $state(false);

const cost = $derived(current ? napkinCost(current) : null);
const baseCost = $derived(
  fixture === "attention" && attentionBase
    ? napkinCost(attentionBase)
    : fixture === "matmul" && matmulBase
      ? napkinCost(matmulBase)
      : null,
);
const projection = $derived(
  current
    ? deriveProjection(current)
    : { ok: false, lines: [], reason: "Loading." },
);
const roundTrips = $derived(
  current
    ? JSON.stringify(fromDiagram(toDiagram(current))) ===
        JSON.stringify(current)
    : false,
);
const trace = $derived([
  ...(baseCost ? [{ label: "base term", cost: baseCost }] : []),
  ...history.map((entry) => ({ label: entry.label, cost: entry.cost })),
]);
const tracePoints = $derived.by(() => {
  if (!trace.length) return "";
  const values = trace.map((point) => point.cost.transferByLevel.l1);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values
    .map((value, index) => {
      const x = values.length === 1 ? 50 : (index / (values.length - 1)) * 100;
      const y = 28 - ((value - min) / span) * 24;
      return `${x},${y}`;
    })
    .join(" ");
});

onMount(async () => {
  try {
    const [attentionResponse, matmulResponse] = await Promise.all([
      fetch("/data/ncd/attention-naive.term.json"),
      fetch("/data/ncd/tiled-matmul.term.json"),
    ]);
    if (!attentionResponse.ok || !matmulResponse.ok)
      throw new Error("NCD term assets failed to load.");
    attentionBase = (await attentionResponse.json()) as NcdTerm;
    matmulBase = (await matmulResponse.json()) as NcdTerm;
    resetTo("attention");
  } catch (error) {
    loadError = error instanceof Error ? error.message : String(error);
  }
});

function controlClass(active = false): string {
  return `inline-flex h-control items-center justify-center gap-1 border px-1.5 type-button focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50 ${
    active
      ? "border-primary bg-primary/12 text-primary-accent"
      : "border-border bg-card text-foreground hover:bg-muted active:bg-border/50"
  }`;
}

function resetTo(nextFixture: "attention" | "matmul" = fixture): void {
  fixture = nextFixture;
  const base = nextFixture === "attention" ? attentionBase : matmulBase;
  if (!base) return;
  current = cloneTerm(base);
  history = [];
  redoStack = [];
  faStep = 0;
  previewTerm = null;
  jam = null;
  equivalence = null;
  activeGesture = null;
  refusal = `Restored ${base.name}.`;
}

function showEquivalence(before: NcdTerm, after: NcdTerm, label: string): void {
  if (equivalenceTimer) clearTimeout(equivalenceTimer);
  equivalence = { before, after, label, nonce: ++equivalenceNonce };
  equivalenceTimer = setTimeout(() => {
    equivalence = null;
  }, 1050);
}

function refuse(reason: string, target: Omit<SurfaceJam, "reason">): void {
  previewTerm = null;
  jam = { ...target, reason };
  refusal = `Refused: ${reason}`;
}

function commit(label: string, move: NcdMove): void {
  if (!current) return;
  try {
    const before = cloneTerm(current);
    const next = applyMove(current, move);
    const entry: NcdHistoryEntry = {
      label,
      forward: move,
      inverse: inverseMove(move),
      termHash: termHash(next),
      cost: napkinCost(next),
    };
    current = next;
    previewTerm = null;
    jam = null;
    history = [...history, entry];
    redoStack = [];
    refusal = `${label} applied as an invertible term relabeling.`;
    showEquivalence(before, next, label);
  } catch (error) {
    refusal = `Refused: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function undo(): void {
  const entry = history.at(-1);
  if (!entry || !current) return;
  const before = cloneTerm(current);
  current = applyMove(current, entry.inverse);
  history = history.slice(0, -1);
  redoStack = [...redoStack, entry];
  faStep = Math.min(faStep, history.length);
  refusal = `Undo applied inverse ${entry.inverse.op} relabeling.`;
  previewTerm = null;
  jam = null;
  showEquivalence(before, current, `undo · ${entry.label}`);
}

function redo(): void {
  const entry = redoStack.at(-1);
  if (!entry || !current) return;
  const before = cloneTerm(current);
  current = applyMove(current, entry.forward);
  redoStack = redoStack.slice(0, -1);
  history = [...history, entry];
  refusal = `Redo applied ${entry.forward.op} relabeling.`;
  previewTerm = null;
  jam = null;
  showEquivalence(before, current, `redo · ${entry.label}`);
}

function attemptPartition(axisId: string, kind: PartitionKind): void {
  if (!current) return;
  const axis = axisById(current, axisId);
  const size = kind === "group" ? groupSize : streamSize;
  const after: PartitionDecoration = {
    axisId,
    kind,
    size,
    label: `${kind === "group" ? "g" : "s"}_${axis.label}`,
  };
  const legal = partitionLegality(current, after);
  if (!legal.legal) {
    const blockedBox =
      kind === "stream"
        ? current.semantic.boxes.find(
            (box) => box.streamability.kind === "none",
          )
        : undefined;
    refuse(
      legal.reason,
      blockedBox
        ? { target: "box", id: blockedBox.id }
        : { target: "axis", id: axisId },
    );
    return;
  }
  commit(`${kind}-partition ${axis.label}=${size}`, {
    op: "partition",
    axisId,
    before: current.decorations.partitions.find(
      (item) => item.axisId === axisId,
    ),
    after,
  });
}

function previewPartition(axisId: string, kind: PartitionKind): void {
  if (!current) return;
  const axis = axisById(current, axisId);
  const size = kind === "group" ? groupSize : streamSize;
  const after: PartitionDecoration = {
    axisId,
    kind,
    size,
    label: `${kind === "group" ? "g" : "s"}_${axis.label}`,
  };
  if (!partitionLegality(current, after).legal) {
    previewTerm = null;
    return;
  }
  previewTerm = applyMove(current, {
    op: "partition",
    axisId,
    before: current.decorations.partitions.find(
      (item) => item.axisId === axisId,
    ),
    after,
  });
}

function attemptResidency(
  wireId: string,
  column: number,
  level: NcdLevel,
): void {
  if (!current) return;
  const state = current.decorations.residency.find(
    (item) => item.wireId === wireId && item.column === column,
  );
  if (!state) return;
  const legal = recolorLegality(current, wireId, column, level);
  if (!legal.legal) {
    const blockedBox = current.semantic.boxes.find(
      (box) => box.streamability.kind === "none",
    );
    refuse(
      legal.reason,
      legal.reason.includes("softmax") && blockedBox
        ? { target: "box", id: blockedBox.id }
        : { target: "residency", id: wireId, column },
    );
    return;
  }
  commit(`recolor ${wireId}@${column} ${state.level}→${level}`, {
    op: "recolor",
    wireId,
    column,
    before: state.level,
    after: level,
  });
}

function previewResidency(
  wireId: string,
  column: number,
  level: NcdLevel,
): void {
  if (!current) return;
  const state = current.decorations.residency.find(
    (item) => item.wireId === wireId && item.column === column,
  );
  if (!state || !recolorLegality(current, wireId, column, level).legal) {
    previewTerm = null;
    return;
  }
  previewTerm = applyMove(current, {
    op: "recolor",
    wireId,
    column,
    before: state.level,
    after: level,
  });
}

function clearPreview(): void {
  previewTerm = null;
}

function admitLemma(): void {
  if (!current) return;
  if (current.decorations.admittedLemmas.includes("online-softmax-rescaling")) {
    refuse("online-softmax rescaling is already admitted.", {
      target: "box",
      id: "softmax",
    });
    return;
  }
  commit("admit online-softmax rescaling", onlineSoftmaxLemma(current));
}

function dropLemma(boxId: string): void {
  if (boxId !== "softmax") {
    refuse(
      "This lemma rewrites softmax only; the selected function does not match.",
      {
        target: "box",
        id: boxId,
      },
    );
    return;
  }
  admitLemma();
}

function nextFaStep(): void {
  if (!attentionBase) return;
  if (
    fixture !== "attention" ||
    faStep >= FA_STEPS.length ||
    history.length !== faStep
  ) {
    resetTo("attention");
  }
  if (!current) return;
  try {
    const before = cloneTerm(current);
    const result = applyFaStep(current, faStep);
    current = result.term;
    history = [...history, result.entry];
    redoStack = [];
    refusal = `Walkthrough ${faStep + 1}/${FA_STEPS.length}: ${result.entry.label}`;
    faStep += 1;
    jam = null;
    previewTerm = null;
    showEquivalence(before, current, result.entry.label);
  } catch (error) {
    refusal = `Refused: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function deriveAll(): Promise<void> {
  if (replayRunning) return;
  replayRunning = true;
  resetTo("attention");
  if (!attentionBase) {
    replayRunning = false;
    return;
  }
  try {
    let next = cloneTerm(attentionBase);
    const entries: NcdHistoryEntry[] = [];
    for (let index = 0; index < FA_STEPS.length; index += 1) {
      const before = cloneTerm(next);
      const result = applyFaStep(next, index);
      next = result.term;
      entries.push(result.entry);
      current = next;
      history = [...entries];
      redoStack = [];
      faStep = index + 1;
      refusal = `Replay ${index + 1}/${FA_STEPS.length}: ${result.entry.label}`;
      showEquivalence(before, next, result.entry.label);
      await delay(760);
    }
    refusal =
      "FlashAttention derivation replayed: lemma → fuse → fuse → tile → stream.";
  } finally {
    replayRunning = false;
  }
}

function dragPayload(
  event: DragEvent,
  payload:
    | { type: "partition"; kind: PartitionKind }
    | { type: "level"; level: NcdLevel }
    | { type: "lemma"; lemmaId: string },
): void {
  document.getSelection()?.removeAllRanges();
  event.stopPropagation();
  event.dataTransfer?.setData(
    "application/x-torchlette-ncd",
    JSON.stringify(payload),
  );
  if (event.dataTransfer) {
    event.dataTransfer.setData("text/plain", "");
    event.dataTransfer.effectAllowed = "copy";
  }
  activeGesture =
    payload.type === "partition"
      ? {
          type: "partition",
          kind: payload.kind,
          size: payload.kind === "group" ? groupSize : streamSize,
        }
      : payload;
}

function finishGesture(): void {
  activeGesture = null;
  previewTerm = null;
}

function handleGlobalKeydown(event: KeyboardEvent): void {
  if (event.key === "?" && !event.repeat) {
    event.preventDefault();
    helpOpen = !helpOpen;
    return;
  }
  if (event.key === "Escape") {
    helpOpen = false;
    activeGesture = null;
    previewTerm = null;
    jam = null;
  }
}

function formatElements(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(3)} M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(3)} K`;
  return value.toLocaleString();
}
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

{#if loadError}
  <main class="pad-box stack-field min-h-0 flex-1 overflow-y-auto">
    <h1 class="type-heading">NCD assets failed to load</h1>
    <p class="prose text-destructive-strong">{loadError}</p>
  </main>
{:else if !current || !cost}
  <main class="pad-box min-h-0 flex-1 type-body">Loading editable NCD terms…</main>
{:else}
  <main class="stack-field min-h-0 flex-1 overflow-y-auto bg-background">
    <section class="stack-field border-b border-border pad-box">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div class="stack-tight">
          <h1 class="type-heading">Editable Neural Circuit Diagram</h1>
          <p class="type-body text-muted-foreground">Semantics stay fixed; tiling, streaming, and fusion are labels on one term.</p>
        </div>
        <div class="flex items-center gap-1">
          <button class={controlClass(fixture === "attention")} onclick={() => resetTo("attention")}>Attention</button>
          <button class={controlClass(fixture === "matmul")} onclick={() => resetTo("matmul")}>Matmul</button>
          {#if fixture === "attention"}
            <button class={controlClass(true)} disabled={replayRunning} onclick={deriveAll}><Play size={11} /> {replayRunning ? "Replaying FA…" : "Derive FlashAttention"}</button>
          {/if}
          <button class={controlClass()} disabled={!history.length} onclick={undo}><Undo2 size={11} /> Undo</button>
          <button class={controlClass()} disabled={!redoStack.length} onclick={redo}><Redo2 size={11} /> Redo</button>
          <button class={controlClass()} onclick={() => resetTo()}><RotateCcw size={11} /> Reset</button>
          <button class={controlClass(helpOpen)} aria-label="Open gesture help" aria-pressed={helpOpen} onclick={() => (helpOpen = !helpOpen)}><CircleHelp size={12} /> ?</button>
        </div>
      </div>
      <div class="grid grid-cols-[auto_1fr_auto] items-center gap-1 border border-border bg-card pad-box">
        <span class={roundTrips ? "type-tag text-success" : "type-tag text-destructive-strong"}>{roundTrips ? "TERM ≡ DIAGRAM ≡ TERM" : "ROUND-TRIP FAILURE"}</span>
        <span class={refusal.startsWith("Refused") ? "type-body text-destructive-strong" : "type-body text-muted-foreground"}>{refusal}</span>
        <span class="type-value">{termHash(current)}</span>
      </div>
    </section>

    <section class="stack-field border-b border-border pad-box">
      <div class="stack-tight">
        <div class="flex items-center justify-between gap-2">
          <span class="type-label">Gesture palette</span>
          <span class="type-fine text-muted-foreground">Drag a chip to a highlighted target; select a brush and press a region to paint.</span>
        </div>
        <div class="grid grid-cols-[minmax(13rem,1.3fr)_repeat(4,minmax(10rem,1fr))] border border-border max-[1100px]:grid-cols-2">
        <div class="flex items-center gap-2 border-r border-border bg-card pad-box" aria-label="Memory level graph">
          <svg class="h-14 w-24" viewBox="0 0 96 56" aria-label="WGSL memory level graph">
            <line x1="18" y1="13" x2="47" y2="29" stroke="currentColor" stroke-width="1" />
            <line x1="47" y1="29" x2="28" y2="47" stroke="currentColor" stroke-width="1" />
            <line x1="47" y1="29" x2="73" y2="47" stroke="currentColor" stroke-width="1" />
            <circle cx="18" cy="13" r="7" class="fill-foreground" />
            <circle cx="47" cy="29" r="7" style="fill:var(--ncd-level-one-ink)" />
            <circle cx="28" cy="47" r="6" class="fill-success" />
            <circle cx="73" cy="47" r="6" class="fill-primary" />
          </svg>
          <div class="stack-tight">
            <span class="type-label">Level graph</span>
            <span class="type-fine text-muted-foreground">ℓ0 global → ℓ1 workgroup → register / invocation</span>
          </div>
        </div>
        <div class="stack-tight border-r border-border bg-card pad-box">
          <span class="type-label">Tile an axis</span>
          <div class="flex items-center gap-1">
            <button class={controlClass(activeGesture?.type === "partition" && activeGesture.kind === "group")} draggable="true" ondragstart={(event) => dragPayload(event, { type: "partition", kind: "group" })} ondragend={finishGesture}>gₐ · Group</button>
            <input class="h-control min-w-0 flex-1 border border-input bg-card px-1 type-value" type="number" min="1" bind:value={groupSize} aria-label="Group size" />
          </div>
          <span class="type-fine text-muted-foreground">Drop on a divisible axis.</span>
        </div>
        <div class="stack-tight border-r border-border bg-card pad-box">
          <span class="type-label">Stream an axis</span>
          <div class="flex items-center gap-1">
            <button class={controlClass(activeGesture?.type === "partition" && activeGesture.kind === "stream")} draggable="true" ondragstart={(event) => dragPayload(event, { type: "partition", kind: "stream" })} ondragend={finishGesture}>sₐ · Stream</button>
            <input class="h-control min-w-0 flex-1 border border-input bg-card px-1 type-value" type="number" min="1" bind:value={streamSize} aria-label="Stream size" />
          </div>
          <span class="type-fine text-muted-foreground">Only head/body axes light up.</span>
        </div>
        <div class="stack-tight border-r border-border bg-card pad-box">
          <span class="type-label">Paint residency</span>
          <div class="flex items-center gap-1">
            <button class={controlClass(paintLevel === "l0")} draggable="true" aria-pressed={paintLevel === "l0"} onclick={() => (paintLevel = "l0")} ondragstart={(event) => dragPayload(event, { type: "level", level: "l0" })} ondragend={finishGesture}>ℓ0 · Global</button>
            <button class={controlClass(paintLevel === "l1")} draggable="true" aria-pressed={paintLevel === "l1"} onclick={() => (paintLevel = "l1")} ondragstart={(event) => dragPayload(event, { type: "level", level: "l1" })} ondragend={finishGesture}>ℓ1 · Lower</button>
          </div>
          <span class="type-fine text-muted-foreground">Select, then press a colored region.</span>
        </div>
        <div class="stack-tight bg-card pad-box">
          <span class="type-label">Rewrite a function</span>
          {#if fixture === "attention"}
            <button class={controlClass(activeGesture?.type === "lemma")} draggable="true" ondragstart={(event) => dragPayload(event, { type: "lemma", lemmaId: "online-softmax-rescaling" })} ondragend={finishGesture} onclick={admitLemma}>Lemma · online softmax</button>
            <span class="type-fine text-muted-foreground">Drop on the matching σ box.</span>
          {:else}
            <span class="type-fine text-muted-foreground">No admitted lemma for this term.</span>
          {/if}
        </div>
        </div>
      </div>

      <NcdRenderer
        term={current}
        {previewTerm}
        {paintLevel}
        {jam}
        {equivalence}
        {activeGesture}
        onPartitionDrop={attemptPartition}
        onPartitionPreview={previewPartition}
        onResidencyDrop={attemptResidency}
        onResidencyPreview={previewResidency}
        onLemmaDrop={dropLemma}
        onPreviewClear={clearPreview}
      />
    </section>

    <section class="grid grid-cols-[minmax(18rem,0.8fr)_minmax(20rem,1.2fr)] items-start gap-1 border-b border-border pad-box max-[900px]:grid-cols-1">
      <section class="stack-field border border-border">
          <header class="flex items-center justify-between gap-2 border-b border-border bg-panel px-1 py-0.5"><h2 class="type-title">Cost-annotated proof history</h2><span class="type-value">{trace.length}</span></header>
          <div class="stack-field pad-box">
            <div class="grid grid-cols-3 gap-1 border border-border bg-card pad-box">
              <span class="type-label">Current</span>
              <span class="type-value">H₁ {formatElements(cost.transferByLevel.l1)}</span>
              <span class="type-value">M₁ {formatElements(cost.memoryByLevel.l1)}</span>
            </div>
            <svg class="h-8 w-full border border-border bg-card text-primary" viewBox="0 0 100 32" preserveAspectRatio="none" aria-label="NCD transfer history">
              <polyline points={tracePoints} fill="none" stroke="currentColor" stroke-width="1" vector-effect="non-scaling-stroke" />
            </svg>
            <div class="max-h-36 overflow-y-auto border border-border">
              {#each trace as point, index}
                <div class="grid grid-cols-[auto_1fr_auto] gap-1 border-b border-border bg-background pad-box last:border-b-0">
                  <span class="type-value">{index}</span><span class="truncate type-body">{point.label}</span><span class="type-value">H₁ {formatElements(point.cost.transferByLevel.l1)}</span>
                </div>
              {/each}
            </div>
          </div>
      </section>

      <section class="stack-field border border-border">
        <header class="flex items-center justify-between gap-2 border-b border-border bg-panel px-1 py-0.5"><h2 class="type-title">Derived streamable-normal-form projection</h2><span class={projection.ok ? "type-tag text-success" : "type-tag text-destructive-strong"}>{projection.ok ? "DERIVED" : "AMBIGUOUS"}</span></header>
        {#if projection.ok}
          <pre class="max-h-80 overflow-auto bg-card pad-box type-code">{projection.lines.join("\n")}</pre>
        {:else}
          <p class="border border-destructive bg-destructive/10 pad-box type-body text-destructive-strong">{projection.reason}</p>
        {/if}
      </section>
    </section>

    {#if fixture === "attention"}
      <section class="stack-field pad-box" aria-busy={replayRunning}>
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="stack-tight"><h2 class="type-title">FlashAttention by gestures</h2><p class="type-body text-muted-foreground">A replayable proof script over the same semantic boxes and wires.</p></div>
          <div class="flex gap-1">
            <button class={controlClass()} disabled={replayRunning} onclick={nextFaStep}><StepForward size={11} /> Next step</button>
            <button class={controlClass(true)} disabled={replayRunning} onclick={deriveAll}><Play size={11} /> {replayRunning ? "Replaying…" : "Derive FA"}</button>
          </div>
        </div>
        <div class="grid grid-cols-[repeat(auto-fit,minmax(12rem,1fr))] gap-1">
          {#each FA_STEPS as step, index}
            <div class={`border border-border pad-box ${index < faStep ? "bg-primary/12 text-primary-accent" : index === faStep ? "bg-card" : "bg-background text-muted-foreground"}`}>
              <span class="type-value">{index + 1}</span>
              <span class="type-body">{step.label}</span>
            </div>
          {/each}
        </div>
      </section>
    {/if}
  </main>
{/if}

{#if helpOpen}
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-background/70" role="presentation" onclick={() => (helpOpen = false)}>
    <div class="stack-field w-[34rem] max-w-[calc(100vw-2rem)] border border-border bg-card pad-box" role="dialog" tabindex="-1" aria-modal="true" aria-labelledby="ncd-help-title" onclick={(event) => event.stopPropagation()} onkeydown={(event) => event.stopPropagation()}>
      <header class="flex items-center justify-between gap-2 border-b border-border pb-1">
        <h2 id="ncd-help-title" class="type-heading">NCD canvas controls</h2>
        <button class={controlClass()} onclick={() => (helpOpen = false)}>Esc · Close</button>
      </header>
      <div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
        <span class="type-button">Drag empty canvas</span><span class="type-body">Pan with grab/grabbing feedback.</span>
        <span class="type-button">Trackpad / wheel</span><span class="type-body">Pan vertically and horizontally.</span>
        <span class="type-button">Shift + wheel</span><span class="type-body">Pan horizontally.</span>
        <span class="type-button">Ctrl/⌘ + wheel</span><span class="type-body">Zoom toward the pointer; pinch uses the same path.</span>
        <span class="type-button">gₐ / sₐ chip</span><span class="type-body">Drag to a highlighted axis to tile or stream.</span>
        <span class="type-button">ℓ0 / ℓ1 brush</span><span class="type-body">Select a level, then press and release on a valid residency region.</span>
        <span class="type-button">Lemma chip</span><span class="type-body">Drop on the matching function box.</span>
        <span class="type-button">Escape</span><span class="type-body">Cancel the active pan, paint, or drag without committing.</span>
        <span class="type-button">?</span><span class="type-body">Toggle this reference.</span>
      </div>
    </div>
  </div>
{/if}
