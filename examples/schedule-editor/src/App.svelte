<script lang="ts">
import {
  Clipboard,
  Download,
  Minus,
  Plus,
  Redo2,
  RotateCcw,
  Undo2,
} from "@lucide/svelte";
import { onMount } from "svelte";
import NcdLearningGame from "./lib/components/ncd/NcdLearningGame.svelte";
import AppBar from "./lib/components/primitives/AppBar.svelte";
import ThemeProvider from "./lib/components/theme/ThemeProvider.svelte";
import ThemeToggle from "./lib/components/theme/ThemeToggle.svelte";
import Workbench from "./lib/components/workbench/Workbench.svelte";
import { calculateStaticCost, scheduleStateHash } from "./lib/cost-model";
import {
  applyMove,
  boundaryKeys,
  clonePartition,
  hexHash,
  makeMerge,
  makeSplit,
  mergeLegality,
} from "./lib/partition";
import {
  cloneScheduleState,
  type DeviceModel,
  FALLBACK_DEVICE,
  readDeviceModel,
  type ScheduleHistoryPoint,
  type ScheduleState,
} from "./lib/schedule-state";
import type {
  HistoryEntry,
  Partition,
  PartitionMove,
  PlanNode,
  ScheduleDump,
} from "./lib/types";

let dump = $state<ScheduleDump | null>(null);
let basePartition = $state<Partition | null>(null);
let current = $state<Partition | null>(null);
let selectedIslands = $state<number[]>([]);
let selectedNodePos = $state<number | null>(null);
let selectedCut = $state<{ island: number; cut: number } | null>(null);
let undoStack = $state<HistoryEntry[]>([]);
let redoStack = $state<HistoryEntry[]>([]);
let moves = $state<PartitionMove[]>([]);
let zoom = $state(10);
let notice = $state("Ready.");
let loadError = $state("");
let matmulTemplate = $state<ScheduleState | null>(null);
let attentionTemplate = $state<ScheduleState | null>(null);
let scheduleHistory = $state<ScheduleHistoryPoint[]>([]);
let scheduleHistoryCursor = $state(-1);
let scheduleHistoryId = 0;
let deviceModel = $state<DeviceModel>({ ...FALLBACK_DEVICE });
let scheduleBindingNote = $state("");
let mode = $state<"schedule" | "ncd">("schedule");

const nodesByPos = $derived(
  new Map((dump?.nodes ?? []).map((node) => [node.pos, node])),
);
const selectedNode = $derived(
  selectedNodePos === null ? null : (nodesByPos.get(selectedNodePos) ?? null),
);
const baseBoundaries = $derived(
  basePartition ? boundaryKeys(basePartition) : new Set<string>(),
);
const requestedBoundaries = $derived(
  current ? boundaryKeys(current) : new Set<string>(),
);
const matchesDerived = $derived(
  Boolean(
    basePartition &&
      current &&
      basePartition.boundaryHash === current.boundaryHash,
  ),
);
const mergeState = $derived(
  current
    ? mergeLegality(current, selectedIslands, nodesByPos)
    : { legal: false, reason: "Partition is loading." },
);
const exportObject = $derived(
  dump && basePartition && current
    ? {
        schemaVersion: 1,
        type: "schedule.partition.request",
        planFingerprint: dump.meta.planFingerprint,
        baseBoundaryHash: basePartition.boundaryHash,
        requestedPartition: current,
        moves,
      }
    : null,
);
const exportText = $derived(
  exportObject ? JSON.stringify(exportObject, null, 2) : "",
);
const currentSchedule = $derived(
  scheduleHistoryCursor >= 0
    ? (scheduleHistory[scheduleHistoryCursor]?.state ?? null)
    : null,
);
const currentScheduleCost = $derived(
  currentSchedule ? calculateStaticCost(currentSchedule, deviceModel) : null,
);
const structuralCosts = $derived(
  undoStack.flatMap((entry, index) =>
    entry.staticCost
      ? [
          {
            label: `${index + 1}. partition ${entry.forward.op}`,
            cost: entry.staticCost.predictedMs,
          },
        ]
      : [],
  ),
);

onMount(async () => {
  try {
    const [dumpResponse, matmulResponse, attentionResponse, detectedDevice] =
      await Promise.all([
        fetch("/data/gpt2-tiny-forward.json"),
        fetch("/data/schedule-states/tiled-matmul.json"),
        fetch("/data/schedule-states/authored-attention-forward.json"),
        readDeviceModel(),
      ]);
    for (const response of [dumpResponse, matmulResponse, attentionResponse]) {
      if (!response.ok)
        throw new Error(
          `${response.url}: ${response.status} ${response.statusText}`,
        );
    }
    dump = (await dumpResponse.json()) as ScheduleDump;
    matmulTemplate = (await matmulResponse.json()) as ScheduleState;
    attentionTemplate = (await attentionResponse.json()) as ScheduleState;
    deviceModel = detectedDevice;
    basePartition = clonePartition(dump.partition);
    current = clonePartition(dump.partition);
    selectedNodePos = dump.nodes[0]?.pos ?? null;
    notice = `Loaded ${dump.nodes.length} nodes in ${dump.partition.islands.length} islands.`;
  } catch (error) {
    loadError = error instanceof Error ? error.message : String(error);
  }
});

function controlClass(
  tone: "default" | "primary" | "destructive" = "default",
): string {
  const tones = {
    default:
      "border-border bg-card text-foreground hover:bg-muted active:bg-border/50",
    primary:
      "border-primary-accent/55 bg-primary-accent/5 text-primary-accent hover:bg-primary-accent/(--tint-hover) active:bg-primary-accent/(--tint-active)",
    destructive:
      "border-destructive-accent/55 bg-destructive-accent/5 text-destructive-accent hover:bg-destructive-accent/(--tint-hover) active:bg-destructive-accent/(--tint-active)",
  };
  return `inline-flex h-control items-center justify-center gap-1 border px-1.5 type-button focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50 ${tones[tone]}`;
}

function islandClass(kind: string, selected: boolean): string {
  const kindClass =
    kind === "fused"
      ? "border-primary bg-primary/12 text-primary-accent"
      : kind === "reduction"
        ? "border-warning bg-warning/15 text-warning-strong"
        : "border-border-strong bg-card text-card-foreground";
  return `${kindClass} ${selected ? "outline outline-1 outline-ring outline-offset-1" : "hover:bg-muted"}`;
}

function selectIsland(index: number): void {
  selectedCut = null;
  if (selectedIslands.includes(index)) {
    selectedIslands = selectedIslands.filter((item) => item !== index);
  } else if (selectedIslands.length < 2) {
    selectedIslands = [...selectedIslands, index];
  } else {
    selectedIslands = [index];
  }
  const member = current?.islands[index]?.members[0];
  if (member !== undefined) selectedNodePos = member;
  openSchedule(index);
}

function openSchedule(index: number): void {
  const island = current?.islands[index];
  if (!island || !matmulTemplate || !attentionTemplate) return;
  const attentionRegion = island.members.some(
    (member) =>
      (member >= 82 && member <= 107) || (member >= 166 && member <= 189),
  );
  const template = attentionRegion ? attentionTemplate : matmulTemplate;
  const state = cloneScheduleState(template);
  const cost = calculateStaticCost(state, deviceModel);
  scheduleHistoryId += 1;
  scheduleHistory = [
    {
      id: scheduleHistoryId,
      label: "opened island state",
      state,
      stateHash: scheduleStateHash(state),
      cost,
    },
  ];
  scheduleHistoryCursor = 0;
  scheduleBindingNote = attentionRegion
    ? `Island ${index} lies in the decomposed attention region; showing the authored fused-forward target state.`
    : `Island ${index} opens the tiled-matmul consumer fixture for the intra-island proposal.`;
}

function chooseCut(island: number, cut: number): void {
  selectedIslands = [island];
  selectedCut = { island, cut };
  const member = current?.islands[island]?.members[cut];
  if (member !== undefined) selectedNodePos = member;
  notice = "Member boundary selected. Split is available.";
}

function commit(entry: HistoryEntry): void {
  if (!current) return;
  const annotated: HistoryEntry =
    currentScheduleCost && currentSchedule
      ? {
          ...entry,
          staticCost: {
            stateHash: scheduleStateHash(currentSchedule),
            predictedMs: currentScheduleCost.predictedMs,
            arithmeticIntensity: currentScheduleCost.arithmeticIntensity,
            rooflineBound: currentScheduleCost.rooflineBound,
          },
        }
      : entry;
  current = applyMove(current, annotated.forward);
  undoStack = [...undoStack, annotated];
  redoStack = [];
  moves = [...moves, entry.forward];
  selectedIslands = [];
  selectedCut = null;
}

function mergeSelected(): void {
  if (!current || !mergeState.legal || !mergeState.indices) {
    notice = `Refused: ${mergeState.reason}`;
    return;
  }
  commit(makeMerge(current, mergeState.indices[0], mergeState.indices[1]));
  notice = "Merged adjacent islands; inverse split recorded for undo.";
}

function splitSelected(): void {
  if (!current || !selectedCut) {
    notice = "Refused: select an interior member boundary first.";
    return;
  }
  commit(makeSplit(current, selectedCut.island, selectedCut.cut));
  notice = "Split island; inverse merge recorded for undo.";
}

function undo(): void {
  const entry = undoStack.at(-1);
  if (!entry || !current) return;
  current = applyMove(current, entry.inverse);
  undoStack = undoStack.slice(0, -1);
  redoStack = [...redoStack, entry];
  moves = moves.slice(0, -1);
  selectedIslands = [];
  selectedCut = null;
  notice = `Undo applied structurally as ${entry.inverse.op}.`;
}

function redo(): void {
  const entry = redoStack.at(-1);
  if (!entry || !current) return;
  current = applyMove(current, entry.forward);
  redoStack = redoStack.slice(0, -1);
  undoStack = [...undoStack, entry];
  moves = [...moves, entry.forward];
  selectedIslands = [];
  selectedCut = null;
  notice = `Redo applied structurally as ${entry.forward.op}.`;
}

function reset(): void {
  if (!basePartition) return;
  current = clonePartition(basePartition);
  undoStack = [];
  redoStack = [];
  moves = [];
  selectedIslands = [];
  selectedCut = null;
  notice = "Restored the detector-derived partition.";
}

function editSchedule(
  label: string,
  edit: (state: ScheduleState) => void,
): void {
  if (!currentSchedule) return;
  const next = cloneScheduleState(currentSchedule);
  edit(next);
  const cost = calculateStaticCost(next, deviceModel);
  scheduleHistoryId += 1;
  scheduleHistory = [
    ...scheduleHistory.slice(0, scheduleHistoryCursor + 1),
    {
      id: scheduleHistoryId,
      label,
      state: next,
      stateHash: scheduleStateHash(next),
      cost,
    },
  ];
  scheduleHistoryCursor = scheduleHistory.length - 1;
  notice = `Decoration edit recorded: ${label}.`;
}

function undoSchedule(): void {
  if (scheduleHistoryCursor <= 0) return;
  scheduleHistoryCursor -= 1;
  notice = "Schedule decoration undo restored a prior state object.";
}

function redoSchedule(): void {
  if (scheduleHistoryCursor >= scheduleHistory.length - 1) return;
  scheduleHistoryCursor += 1;
  notice = "Schedule decoration redo restored the next state object.";
}

function updateDevice(next: DeviceModel): void {
  deviceModel = next;
  scheduleHistory = scheduleHistory.map((point) => ({
    ...point,
    cost: calculateStaticCost(point.state, next),
  }));
}

async function copyExport(): Promise<void> {
  await navigator.clipboard.writeText(exportText);
  notice = "Requested partition JSON copied.";
}

function downloadExport(): void {
  const url = URL.createObjectURL(
    new Blob([exportText], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "requested-partition.json";
  anchor.click();
  URL.revokeObjectURL(url);
  notice = "Requested partition JSON exported.";
}

function boundaryTone(left: number, right: number): string {
  const key = `${left}|${right}`;
  if (requestedBoundaries.has(key) && !baseBoundaries.has(key))
    return "bg-success";
  return "bg-border-strong";
}
</script>

{#if mode === "ncd"}
  <NcdLearningGame onExit={() => (mode = "schedule")} />
{:else}
<ThemeProvider>
  <div class="fixed inset-0 flex flex-col overflow-hidden bg-background text-foreground">
    <AppBar title="torchlette" context="Schedule editor">
      <ThemeToggle integrated />
    </AppBar>

    <nav class="flex h-control shrink-0 items-stretch border-b border-border bg-panel" aria-label="Editor mode">
      <button
        class="border-r border-border bg-primary/12 px-2 type-body text-primary-accent"
        onclick={() => (mode = "schedule")}
      >Island schedule</button>
      <button
        class="border-r border-border px-2 type-body hover:bg-muted active:bg-border/50"
        data-game-affordance="navigation"
        data-action-id="enter-game"
        onclick={() => (mode = "ncd")}
      >NCD diagram</button>
    </nav>

    {#if loadError}
      <main class="pad-box stack-field min-h-0 flex-1 overflow-y-auto">
        <h1 class="type-heading">Ground-truth plan failed to load</h1>
        <p class="prose text-destructive-strong">{loadError}</p>
      </main>
    {:else if !dump || !current || !basePartition}
      <main class="pad-box min-h-0 flex-1 type-body">Loading schedule data…</main>
    {:else}
      <main class="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_18rem] overflow-hidden max-[880px]:grid-cols-1">
        <div class="stack-field min-h-0 overflow-y-auto border-r border-border max-[880px]:border-r-0">
          <section class="stack-field border-b border-border pad-box">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="stack-tight">
                <h1 class="type-heading">Emission-order partition</h1>
                <p class="type-body text-muted-foreground">
                  {dump.meta.model} · {dump.meta.step} · {dump.nodes.length} nodes
                </p>
              </div>
              <div class="flex flex-wrap items-center gap-1">
                <button class={controlClass("primary")} disabled={!mergeState.legal} onclick={mergeSelected}>Merge</button>
                <button class={controlClass("primary")} disabled={!selectedCut} onclick={splitSelected}>Split</button>
                <button class={controlClass()} disabled={!undoStack.length} onclick={undo} aria-label="Undo">
                  <Undo2 size={11} /> Undo
                </button>
                <button class={controlClass()} disabled={!redoStack.length} onclick={redo} aria-label="Redo">
                  <Redo2 size={11} /> Redo
                </button>
                <button class={controlClass()} disabled={matchesDerived} onclick={reset} aria-label="Reset to derived">
                  <RotateCcw size={11} /> Reset
                </button>
              </div>
            </div>

            <div class="grid grid-cols-[auto_1fr] gap-1 border border-border bg-card pad-box">
              <span class="type-label">Gesture status</span>
              <span class={mergeState.legal ? "type-body text-success" : "type-body text-muted-foreground"}>
                {selectedCut ? "Split boundary ready." : mergeState.reason}
              </span>
              <span class="type-label">Last action</span>
              <span class="type-body text-muted-foreground">{notice}</span>
            </div>
          </section>

          <section class="stack-field border-b border-border pad-box">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="flex flex-wrap items-center gap-2">
                <span class="flex items-center gap-1 type-body"><span class="h-2 w-2 border border-border-strong bg-card"></span>Sequential</span>
                <span class="flex items-center gap-1 type-body"><span class="h-2 w-2 border border-primary bg-primary/12"></span>Fused</span>
                <span class="flex items-center gap-1 type-body"><span class="h-2 w-2 border border-warning bg-warning/15"></span>Reduction</span>
                <span class="flex items-center gap-1 type-body"><span class="h-3 w-px bg-success"></span>Added boundary</span>
                <span class="flex items-center gap-1 type-body"><span class="h-3 w-px bg-destructive"></span>Removed boundary</span>
              </div>
              <div class="flex items-center gap-1">
                <button class={controlClass()} onclick={() => (zoom = Math.max(4, zoom - 2))} aria-label="Zoom out"><Minus size={11} /></button>
                <span class="w-12 text-center type-value">{zoom}px</span>
                <button class={controlClass()} onclick={() => (zoom = Math.min(28, zoom + 2))} aria-label="Zoom in"><Plus size={11} /></button>
              </div>
            </div>

            <div class="overflow-x-auto border border-border bg-background pad-box" aria-label="Partition lane">
              <div class="flex min-w-max items-stretch gap-1 py-1">
                {#each current.islands as island, islandIndex}
                  {@const width = Math.max(22, island.members.length * zoom)}
                  <div class="relative flex items-stretch">
                    <button
                      class={`relative flex h-16 flex-col justify-between overflow-hidden border pad-box text-left ${islandClass(island.kind, selectedIslands.includes(islandIndex))}`}
                      style={`width:${width}px`}
                      onclick={() => selectIsland(islandIndex)}
                      title={`${island.kind}: ${island.members.length} member${island.members.length === 1 ? "" : "s"}`}
                    >
                      <span class="truncate type-tag">{island.kind}</span>
                      {#if zoom >= 12}
                        <span class="truncate type-value">{island.members[0]}–{island.members.at(-1)}</span>
                      {/if}
                      <span class="type-value">{island.members.length}</span>
                      {#each island.members.slice(0, -1) as member, memberIndex}
                        {@const next = island.members[memberIndex + 1]}
                        {#if baseBoundaries.has(`${member}|${next}`) && !requestedBoundaries.has(`${member}|${next}`)}
                          <span
                            class="absolute inset-y-0 w-px bg-destructive"
                            style={`left:${((memberIndex + 1) / island.members.length) * 100}%`}
                            title={`Removed detector boundary ${member}|${next}`}
                          ></span>
                        {/if}
                      {/each}
                    </button>
                    {#if islandIndex < current.islands.length - 1}
                      {@const nextIsland = current.islands[islandIndex + 1]}
                      <span class={`w-px shrink-0 ${boundaryTone(island.members.at(-1)!, nextIsland.members[0])}`}></span>
                    {/if}
                  </div>
                {/each}
              </div>
            </div>
            <p class="type-fine text-muted-foreground">
              Width is proportional to member count. Scroll horizontally or change zoom; click an island to open its intra-island ScheduleState.
            </p>
          </section>

          {#if currentSchedule && currentScheduleCost}
            <Workbench
              state={currentSchedule}
              cost={currentScheduleCost}
              device={deviceModel}
              history={scheduleHistory}
              historyCursor={scheduleHistoryCursor}
              {structuralCosts}
              bindingNote={scheduleBindingNote}
              onEdit={editSchedule}
              onUndo={undoSchedule}
              onRedo={redoSchedule}
              onDeviceChange={updateDevice}
            />
          {:else}
            <section class="stack-field border-b border-border pad-box">
              <h2 class="type-title">Intra-island workbench</h2>
              <p class="prose text-muted-foreground">Select an island in the lane to zoom into its skeleton, decorations, and performance model.</p>
            </section>
          {/if}

          <section class="stack-field pad-box">
            <div class="flex items-center justify-between gap-2">
              <h2 class="type-title">Requested partition contract</h2>
              <div class="flex gap-1">
                <button class={controlClass()} onclick={copyExport}><Clipboard size={11} /> Copy</button>
                <button class={controlClass()} onclick={downloadExport}><Download size={11} /> Export</button>
              </div>
            </div>
            <pre class="max-h-72 overflow-auto border border-border bg-card pad-box type-code">{exportText}</pre>
          </section>
        </div>

        <aside class="stack-field min-h-0 overflow-y-auto bg-background pad-box max-[880px]:hidden">
          <section class="stack-field border border-border">
            <header class="flex items-center justify-between gap-2 border-b border-border bg-panel px-1 py-0.5">
              <h2 class="type-title">Partition identity</h2>
              <span class={matchesDerived ? "type-tag text-success" : "type-tag text-warning-strong"}>
                {matchesDerived ? "DERIVED" : "MODIFIED"}
              </span>
            </header>
            <dl class="grid grid-cols-[1fr_auto] gap-1 pad-box">
              <dt class="type-label">Detector hash</dt><dd class="type-value">{hexHash(basePartition.boundaryHash)}</dd>
              <dt class="type-label">Current hash</dt><dd class="type-value">{hexHash(current.boundaryHash)}</dd>
              <dt class="type-label">Islands</dt><dd class="type-value">{current.islands.length}</dd>
              <dt class="type-label">Moves</dt><dd class="type-value">{moves.length}</dd>
            </dl>
          </section>

          <section class="stack-field border border-border">
            <header class="border-b border-border bg-panel px-1 py-0.5"><h2 class="type-title">Island members</h2></header>
            {#if selectedIslands.length === 1}
              {@const islandIndex = selectedIslands[0]}
              {@const island = current.islands[islandIndex]}
              <div class="stack-field pad-box">
                <div class="flex items-center justify-between gap-2">
                  <span class="type-label">Island {islandIndex}</span>
                  <span class="type-tag">{island.kind} · {island.members.length}</span>
                </div>
                <div class="flex flex-wrap items-center gap-1">
                  {#each island.members as member, memberIndex}
                    <button
                      class={`${controlClass()} ${selectedNodePos === member ? "bg-primary/12 text-primary-accent" : ""}`}
                      onclick={() => (selectedNodePos = member)}
                    >{member}</button>
                    {#if memberIndex < island.members.length - 1}
                      <button
                        class={`h-control-sm w-2 border-x border-border bg-card hover:bg-muted active:bg-border/50 ${selectedCut?.island === islandIndex && selectedCut.cut === memberIndex + 1 ? "bg-primary/12" : ""}`}
                        title={`Split after member ${member}`}
                        aria-label={`Select split after member ${member}`}
                        onclick={() => chooseCut(islandIndex, memberIndex + 1)}
                      ></button>
                    {/if}
                  {/each}
                </div>
                {#if island.members.length < 2}
                  <p class="type-body text-muted-foreground">No interior member boundary; this island cannot split.</p>
                {:else}
                  <p class="type-fine text-muted-foreground">Select a narrow divider between members, then invoke Split.</p>
                {/if}
              </div>
            {:else}
              <p class="pad-box type-body text-muted-foreground">Select one island to inspect members and choose a split boundary.</p>
            {/if}
          </section>

          <section class="stack-field border border-border">
            <header class="border-b border-border bg-panel px-1 py-0.5"><h2 class="type-title">Node detail</h2></header>
            {#if selectedNode}
              <dl class="grid grid-cols-[auto_1fr] gap-1 pad-box">
                <dt class="type-label">Position</dt><dd class="type-value">{selectedNode.pos}</dd>
                <dt class="type-label">Operation</dt><dd class="type-code">{selectedNode.op}</dd>
                <dt class="type-label">Shape</dt><dd class="type-value">[{selectedNode.shape.join(", ")}]</dd>
                <dt class="type-label">Dtype</dt><dd class="type-tag">{selectedNode.dtype}</dd>
                {#if selectedNode.label}<dt class="type-label">Module</dt><dd class="type-body">{selectedNode.label}</dd>{/if}
              </dl>
            {:else}
              <p class="pad-box type-body text-muted-foreground">Select a member to inspect its node.</p>
            {/if}
          </section>

          <section class="stack-field border border-border">
            <header class="border-b border-border bg-panel px-1 py-0.5"><h2 class="type-title">Legality gate</h2></header>
            <div class="stack-field pad-box">
              <p class="prose">Merges require adjacent, non-reduction, production-fusible, equal-shape islands. Splits accept every interior member boundary.</p>
              <p class="type-body text-muted-foreground">Unprovable device binding limits and atom epilogues are refused by omission; see README.</p>
            </div>
          </section>
        </aside>
      </main>
    {/if}
  </div>
</ThemeProvider>
{/if}
