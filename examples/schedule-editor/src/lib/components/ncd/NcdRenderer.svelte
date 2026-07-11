<script lang="ts">
import { axisById } from "../../ncd/model";
import type { SurfaceEquivalence, SurfaceJam } from "../../ncd/surface-layout";
import {
  orderedResidencies,
  residencyAt,
  surfaceColumnCosts,
  surfaceColumnX,
  surfaceLanes,
  surfaceWorldSize,
  transitionKind,
  wireExpression,
} from "../../ncd/surface-layout";
import type { NcdBox, NcdLevel, NcdTerm, PartitionKind } from "../../ncd/types";

type Props = {
  term: NcdTerm;
  previewTerm?: NcdTerm | null;
  paintLevel: NcdLevel;
  jam?: SurfaceJam | null;
  equivalence?: SurfaceEquivalence | null;
  onPartitionDrop: (axisId: string, kind: PartitionKind) => void;
  onPartitionPreview: (axisId: string, kind: PartitionKind) => void;
  onResidencyDrop: (wireId: string, column: number, level: NcdLevel) => void;
  onResidencyPreview: (wireId: string, column: number, level: NcdLevel) => void;
  onLemmaDrop: (boxId: string) => void;
  onPreviewClear: () => void;
};

let {
  term,
  previewTerm = null,
  paintLevel,
  jam = null,
  equivalence = null,
  onPartitionDrop,
  onPartitionPreview,
  onResidencyDrop,
  onResidencyPreview,
  onLemmaDrop,
  onPreviewClear,
}: Props = $props();

let zoom = $state(0.72);
let panX = $state(18);
let panY = $state(12);
let dragging = $state(false);
let pointerStart = $state({ x: 0, y: 0, panX: 0, panY: 0 });

const lanes = $derived(surfaceLanes(term));
const world = $derived(surfaceWorldSize(term));
const costs = $derived(surfaceColumnCosts(previewTerm ?? term));
const committedCosts = $derived(surfaceColumnCosts(term));
const orderedColumns = $derived(
  [...term.semantic.columns].sort((a, b) => a.index - b.index),
);

function laneFor(wireId: string) {
  return lanes.find((lane) => lane.wire.id === wireId);
}

function partition(axisId: string) {
  return (previewTerm ?? term).decorations.partitions.find(
    (item) => item.axisId === axisId,
  );
}

function divisibility(axisId: string) {
  return term.decorations.divisibility.find((item) => item.axisId === axisId);
}

function isWeaved(axisId: string): boolean {
  return (
    term.semantic.wires.filter((wire) => wire.axisIds.includes(axisId)).length >
    1
  );
}

function boxPosition(box: NcdBox) {
  const connected = [...box.inputWireIds, ...box.outputWireIds]
    .map((wireId) => laneFor(wireId))
    .filter((lane) => lane !== undefined);
  const center =
    connected.reduce((sum, lane) => sum + lane.y + lane.height / 2, 0) /
    Math.max(connected.length, 1);
  return { x: surfaceColumnX(term, box.column) - 80, y: center - 27 };
}

function connectionPath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): string {
  const bend = Math.max(36, Math.abs(x2 - x1) * 0.48);
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
}

function formatCost(value: number): string {
  if (!value) return "—";
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}m`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return value.toLocaleString();
}

function parsePayload(
  event: DragEvent,
):
  | { type: "partition"; kind: PartitionKind }
  | { type: "level"; level: NcdLevel }
  | { type: "lemma"; lemmaId: string }
  | null {
  try {
    return JSON.parse(
      event.dataTransfer?.getData("application/x-torchlette-ncd") ?? "",
    ) as
      | { type: "partition"; kind: PartitionKind }
      | { type: "level"; level: NcdLevel }
      | { type: "lemma"; lemmaId: string };
  } catch {
    return null;
  }
}

function previewAxis(event: DragEvent, axisId: string): void {
  event.preventDefault();
  const payload = parsePayload(event);
  if (payload?.type === "partition") onPartitionPreview(axisId, payload.kind);
}

function dropBox(event: DragEvent, boxId: string): void {
  event.preventDefault();
  event.stopPropagation();
  const payload = parsePayload(event);
  if (payload?.type === "lemma") onLemmaDrop(boxId);
}

function jamMatches(
  target: SurfaceJam["target"],
  id: string,
  column?: number,
): boolean {
  return Boolean(
    jam &&
      jam.target === target &&
      jam.id === id &&
      (column === undefined || jam.column === column),
  );
}

function miniAxes(candidate: NcdTerm): string[] {
  return candidate.semantic.wires
    .slice(0, 4)
    .map((wire) =>
      wire.axisIds.map((axisId) => axisById(candidate, axisId).label).join("·"),
    );
}

function dropAxis(event: DragEvent, axisId: string): void {
  event.preventDefault();
  event.stopPropagation();
  const payload = parsePayload(event);
  if (payload?.type === "partition") onPartitionDrop(axisId, payload.kind);
}

function dropResidency(event: DragEvent, wireId: string, column: number): void {
  event.preventDefault();
  event.stopPropagation();
  const payload = parsePayload(event);
  if (payload?.type === "level") onResidencyDrop(wireId, column, payload.level);
}

function beginPan(event: PointerEvent): void {
  if ((event.target as HTMLElement).closest("[data-ncd-target]")) return;
  dragging = true;
  pointerStart = {
    x: event.clientX,
    y: event.clientY,
    panX,
    panY,
  };
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}

function movePan(event: PointerEvent): void {
  if (!dragging) return;
  panX = pointerStart.panX + event.clientX - pointerStart.x;
  panY = pointerStart.panY + event.clientY - pointerStart.y;
}

function endPan(): void {
  dragging = false;
}

function wheelZoom(event: WheelEvent): void {
  event.preventDefault();
  zoom = Math.min(1.6, Math.max(0.42, zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
}
</script>

<div
  class:dragging
  class="ncd-viewport"
  role="application"
  tabindex="-1"
  aria-label="Pannable neural circuit diagram"
  onpointerdown={beginPan}
  onpointermove={movePan}
  onpointerup={endPan}
  onpointercancel={endPan}
  onwheel={wheelZoom}
>
  <div class="ncd-zoom-readout" aria-hidden="true">{Math.round(zoom * 100)}%</div>
  <div
    class="ncd-world"
    style={`width:${world.width}px;height:${world.height}px;transform:translate(${panX}px,${panY}px) scale(${zoom})`}
  >
    <div
      class="ncd-cost-table"
      style={`width:${world.width}px;--ncd-columns:${orderedColumns.length}`}
    >
      <div class="ncd-cost-corner">per column · elements</div>
      {#each orderedColumns as column}
        <div class="ncd-cost-heading">{column.label}</div>
      {/each}
      <div class="ncd-cost-label"><i>M</i><sub>ℓ1</sub></div>
      {#each costs as columnCost, index}
        <div
          class:cost-preview={columnCost.memory.l1 !==
            committedCosts[index]?.memory.l1}
          class="ncd-cost-cell"
        >{formatCost(columnCost.memory.l1)}</div>
      {/each}
      <div class="ncd-cost-label"><i>H</i><sub>ℓ1</sub></div>
      {#each costs as columnCost, index}
        <div
          class:cost-preview={columnCost.transfer.l1 !==
            committedCosts[index]?.transfer.l1}
          class="ncd-cost-cell"
        >
          {formatCost(columnCost.transfer.l1)}
          {#if index === costs.length - 1}
            <span class="ncd-cumulative">Σ {formatCost(columnCost.cumulative.l1)}</span>
          {/if}
        </div>
      {/each}
    </div>

    <svg
      class="ncd-flow-layer"
      width={world.width}
      height={world.height}
      viewBox={`0 0 ${world.width} ${world.height}`}
      aria-hidden="true"
    >
      {#each lanes as lane}
        {@const states = orderedResidencies(term, lane.wire.id)}
        {#each states.slice(1) as state, stateIndex}
          {@const previous = states[stateIndex]}
          <path
            class:lower-wire={state.level === "l1" || previous.level === "l1"}
            class="ncd-wire-path"
            d={connectionPath(
              surfaceColumnX(term, previous.column) + 58,
              lane.y + lane.height / 2,
              surfaceColumnX(term, state.column) - 6,
              lane.y + lane.height / 2,
            )}
          />
        {/each}
      {/each}

      {#each term.semantic.boxes as box}
        {@const position = boxPosition(box)}
        {#each box.inputWireIds as wireId, inputIndex}
          {@const lane = laneFor(wireId)}
          {#if lane}
            <path
              class="ncd-function-path"
              d={connectionPath(
                surfaceColumnX(term, Math.max(0, box.column - 1)) + 58,
                lane.y + lane.height / 2,
                position.x,
                position.y + 15 + inputIndex * 9,
              )}
            />
          {/if}
        {/each}
        {#each box.outputWireIds as wireId, outputIndex}
          {@const lane = laneFor(wireId)}
          {#if lane}
            <path
              class="ncd-function-path"
              d={connectionPath(
                position.x + 64,
                position.y + 22 + outputIndex * 9,
                surfaceColumnX(term, box.column) - 6,
                lane.y + lane.height / 2,
              )}
            />
          {/if}
        {/each}
      {/each}
    </svg>

    {#each lanes as lane, laneIndex}
      {#if lane.wire.tupleGroup && laneIndex > 0 && lanes[laneIndex - 1].wire.tupleGroup === lane.wire.tupleGroup}
        <div
          class="ncd-tuple-rule"
          style={`top:${lane.y - 5}px;width:${world.width - 96}px`}
        ></div>
      {/if}
      {#each orderedResidencies(term, lane.wire.id) as state}
        {@const displayTerm = previewTerm ?? term}
        {@const displayState = residencyAt(displayTerm, lane.wire.id, state.column) ?? state}
        {@const region = transitionKind(displayTerm, lane.wire.id, state.column)}
        <div
          class:jammed={jamMatches("residency", lane.wire.id, state.column)}
          class={`ncd-array ncd-region-${region}`}
          data-ncd-target
          data-wire={lane.wire.id}
          data-column={state.column}
          role="button"
          tabindex="0"
          aria-label={`${lane.wire.label} column ${state.column} residency ${displayState.level}`}
          style={`left:${surfaceColumnX(term, state.column)}px;top:${lane.y}px;height:${lane.height}px`}
          ondragover={(event) => event.preventDefault()}
          ondrop={(event) => dropResidency(event, lane.wire.id, state.column)}
          onpointerenter={() =>
            onResidencyPreview(lane.wire.id, state.column, paintLevel)}
          onpointerleave={onPreviewClear}
          onpointerdown={(event) => {
            if (event.button === 0) {
              event.stopPropagation();
              onResidencyDrop(lane.wire.id, state.column, paintLevel);
            }
          }}
        >
          <span class="ncd-region-caption">
            {region === "load"
              ? "Load to ℓ1"
              : region === "save"
                ? "Save to ℓ0"
                : displayState.level}
          </span>
          <span class="ncd-array-name">{lane.wire.label}</span>
          <span class="ncd-array-expression">{wireExpression(term, lane.wire)}</span>
          <div class="ncd-axis-stack">
            {#each lane.wire.axisIds as axisId}
              {@const axis = axisById(term, axisId)}
              {@const part = partition(axisId)}
              {@const divisor = divisibility(axisId)}
              <div
                class:jammed={jamMatches("axis", axisId)}
                class="ncd-axis"
                data-axis={axisId}
                class:weaved={isWeaved(axisId)}
                role="button"
                tabindex="0"
                aria-label={`${lane.wire.label} axis ${axis.label}`}
                ondragover={(event) => previewAxis(event, axisId)}
                ondragleave={onPreviewClear}
                ondrop={(event) => dropAxis(event, axisId)}
              >
                <i>{axis.label}</i>
                <span class="ncd-axis-rule"></span>
                <span class="ncd-axis-size">
                  {displayState.level === "l1" && part ? part.size : axis.size}
                  {#if divisor}<sup>∣{divisor.multiple}</sup>{/if}
                </span>
                {#if part}
                  <span class={`ncd-partition ncd-partition-${part.kind}`}>
                    <i>{part.kind === "group" ? "g" : "s"}</i><sub>{axis.label}</sub>
                  </span>
                {/if}
                {#if jamMatches("axis", axisId)}
                  <span class="ncd-jam-note">{jam?.reason}</span>
                {/if}
              </div>
            {/each}
          </div>
          {#if jamMatches("residency", lane.wire.id, state.column)}
            <span class="ncd-jam-note">{jam?.reason}</span>
          {/if}
        </div>
      {/each}
    {/each}

    {#each term.semantic.boxes as box}
      {@const position = boxPosition(box)}
      <div
        class:opaque={box.streamability.kind === "none"}
        class:jammed={jamMatches("box", box.id)}
        class={`ncd-function ncd-function-${box.kind}`}
        data-ncd-target
        data-box={box.id}
        role="button"
        tabindex="0"
        style={`left:${position.x}px;top:${position.y}px`}
        title={box.streamability.kind === "none"
          ? box.streamability.reason
          : "Head/body decomposition available"}
        ondragover={(event) => event.preventDefault()}
        ondrop={(event) => dropBox(event, box.id)}
      >
        <span class="ncd-function-glyph">
          {box.kind === "matmul" ? "◁" : box.kind === "online-softmax" ? "↻" : "σ"}
        </span>
        <span class="ncd-function-label">{box.label}</span>
        {#if jamMatches("box", box.id)}
          <span class="ncd-jam-note">{jam?.reason}</span>
        {/if}
      </div>
    {/each}

    {#if equivalence}
      <div class="ncd-equivalence" data-testid="ncd-equivalence">
        <div class="ncd-equivalence-diagram">
          {#each miniAxes(equivalence.before) as axes}<span><i>{axes}</i></span>{/each}
        </div>
        <strong>≡</strong>
        <div class="ncd-equivalence-diagram after">
          {#each miniAxes(equivalence.after) as axes}<span><i>{axes}</i></span>{/each}
        </div>
        <em>{equivalence.label}</em>
      </div>
    {/if}
  </div>
</div>

<style>
  .ncd-viewport {
    position: relative;
    min-height: 680px;
    overflow: hidden;
    cursor: grab;
    background: var(--ncd-paper);
    color: var(--ncd-ink);
    border: 1px solid var(--border);
    touch-action: none;
  }
  .ncd-viewport.dragging { cursor: grabbing; }
  .ncd-world {
    position: absolute;
    transform-origin: 0 0;
    font-family: Georgia, "Times New Roman", serif;
    transition: transform 80ms linear !important;
  }
  .ncd-zoom-readout {
    position: absolute;
    z-index: 10;
    right: 8px;
    bottom: 8px;
    padding: 2px 5px;
    border: 1px solid var(--ncd-rule-faint);
    background: color-mix(in oklab, var(--ncd-paper) 88%, transparent);
    color: var(--ncd-annotation);
    font: 10px/1 ui-monospace, monospace;
  }
  .ncd-cost-table {
    position: absolute;
    top: 20px;
    left: 0;
    display: grid;
    grid-template-columns: 112px repeat(var(--ncd-columns, 7), 176px);
    grid-auto-rows: 32px;
    border-top: 1px solid var(--ncd-ink);
    border-bottom: 1px solid var(--ncd-ink);
  }
  .ncd-cost-corner,
  .ncd-cost-heading,
  .ncd-cost-label,
  .ncd-cost-cell {
    display: flex;
    align-items: center;
    border-bottom: 1px solid var(--ncd-rule-faint);
  }
  .ncd-cost-corner {
    padding-left: 12px;
    color: var(--ncd-annotation);
    font: 10px/1.2 ui-monospace, monospace;
    letter-spacing: 0.03em;
  }
  .ncd-cost-heading {
    justify-content: center;
    padding: 0 8px;
    overflow: hidden;
    color: var(--ncd-annotation);
    font-size: 11px;
    font-style: italic;
    white-space: nowrap;
  }
  .ncd-cost-label {
    padding-left: 28px;
    font-size: 18px;
  }
  .ncd-cost-cell {
    position: relative;
    justify-content: center;
    border-left: 1px solid var(--ncd-rule-faint);
    font-size: 14px;
    font-variant-numeric: tabular-nums;
  }
  .ncd-cost-cell.cost-preview {
    background: color-mix(in oklab, var(--ncd-level-one) 58%, transparent);
    outline: 1px solid var(--ncd-level-one-ink);
    outline-offset: -2px;
  }
  .ncd-cumulative {
    position: absolute;
    right: 7px;
    top: -24px;
    color: var(--ncd-annotation);
    font-size: 11px;
    font-style: italic;
  }
  .ncd-flow-layer { position: absolute; inset: 0; overflow: visible; }
  .ncd-wire-path,
  .ncd-function-path {
    fill: none;
    stroke: var(--ncd-ink);
    stroke-linecap: round;
    vector-effect: non-scaling-stroke;
  }
  .ncd-wire-path { stroke-width: 1.25; }
  .ncd-wire-path.lower-wire {
    stroke: var(--ncd-level-one-ink);
    stroke-width: 1.5;
  }
  .ncd-function-path { stroke-width: 1.05; }
  .ncd-tuple-rule {
    position: absolute;
    left: 84px;
    border-top: 1px dashed var(--ncd-annotation);
    opacity: 0.65;
  }
  .ncd-array {
    position: absolute;
    z-index: 2;
    width: 64px;
    padding: 19px 5px 5px;
    border: 1px dotted transparent;
    cursor: crosshair;
  }
  .ncd-array:hover { border-color: var(--ncd-ink); }
  .ncd-region-load { background: var(--ncd-load); }
  .ncd-region-save { background: var(--ncd-save); }
  .ncd-region-resident { background: var(--ncd-subalgorithm); }
  .ncd-region-global { background: transparent; }
  .ncd-region-caption {
    position: absolute;
    top: 3px;
    left: 4px;
    color: var(--ncd-annotation);
    font-size: 10px;
    white-space: nowrap;
  }
  .ncd-array-name {
    position: absolute;
    top: -14px;
    left: 0;
    font-size: 11px;
    font-style: italic;
  }
  .ncd-array-expression {
    position: absolute;
    right: 2px;
    top: -14px;
    color: var(--ncd-annotation);
    font-size: 9px;
    font-style: italic;
  }
  .ncd-axis-stack { display: flex; flex-direction: column; }
  .ncd-axis {
    position: relative;
    display: flex;
    height: 17px;
    align-items: baseline;
    gap: 3px;
    border-bottom: 1px solid var(--ncd-ink);
    font-size: 14px;
    line-height: 16px;
  }
  .ncd-axis.weaved i { text-decoration: overline; text-decoration-thickness: 1px; }
  .ncd-axis:hover { background: color-mix(in oklab, var(--ncd-level-one) 15%, transparent); }
  .ncd-axis.jammed,
  .ncd-array.jammed,
  .ncd-function.jammed {
    outline: 2px solid var(--ncd-jam);
    outline-offset: 3px;
    animation: ncd-jam 260ms ease-out 2 !important;
  }
  .ncd-axis-rule { flex: 1; }
  .ncd-axis-size {
    color: var(--ncd-annotation);
    font-size: 10px;
    font-style: italic;
  }
  .ncd-axis-size sup { margin-left: 1px; font-size: 7px; }
  .ncd-partition {
    position: absolute;
    right: -24px;
    top: -3px;
    font-size: 11px;
  }
  .ncd-partition-group { color: var(--ncd-group-ink); }
  .ncd-partition-stream { color: var(--ncd-level-one-ink); }
  .ncd-function {
    position: absolute;
    z-index: 3;
    display: flex;
    width: 64px;
    min-height: 54px;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 1.25px solid var(--ncd-ink);
    background: color-mix(in oklab, var(--ncd-paper) 92%, transparent);
  }
  .ncd-function::before {
    position: absolute;
    z-index: -1;
    inset: -10px;
    background: var(--ncd-subalgorithm);
    content: "";
  }
  .ncd-function.opaque {
    border-style: double;
    background-image: repeating-linear-gradient(
      -45deg,
      transparent,
      transparent 4px,
      var(--ncd-rule-faint) 4px,
      var(--ncd-rule-faint) 5px
    );
  }
  .ncd-function-glyph { font-size: 22px; line-height: 1; }
  .ncd-function-label {
    max-width: 58px;
    margin-top: 4px;
    overflow: hidden;
    font-size: 9px;
    font-style: italic;
    text-align: center;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ncd-jam-note {
    position: absolute;
    z-index: 20;
    left: calc(100% + 8px);
    top: 50%;
    width: 210px;
    padding: 6px 8px;
    border: 1px solid var(--ncd-jam);
    background: color-mix(in oklab, var(--ncd-paper) 92%, var(--ncd-jam));
    color: var(--ncd-jam);
    font: 11px/1.35 ui-monospace, monospace;
    transform: translateY(-50%);
  }
  .ncd-jam-note::before {
    position: absolute;
    top: calc(50% - 4px);
    left: -9px;
    width: 8px;
    border-top: 1px solid var(--ncd-jam);
    content: "";
  }
  .ncd-equivalence {
    position: absolute;
    z-index: 40;
    top: 154px;
    left: 50%;
    display: grid;
    grid-template-columns: 132px 36px 132px;
    align-items: center;
    padding: 15px 20px 27px;
    border: 1px solid var(--ncd-ink);
    background: color-mix(in oklab, var(--ncd-paper) 94%, transparent);
    transform: translateX(-50%);
    animation: ncd-equivalence 1050ms ease both !important;
  }
  .ncd-equivalence > strong {
    font-size: 26px;
    font-weight: 400;
    text-align: center;
  }
  .ncd-equivalence > em {
    position: absolute;
    right: 20px;
    bottom: 7px;
    left: 20px;
    color: var(--ncd-annotation);
    font-size: 10px;
    text-align: center;
  }
  .ncd-equivalence-diagram {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 8px;
    border-inline: 5px solid var(--ncd-load);
  }
  .ncd-equivalence-diagram.after { border-color: var(--ncd-save); }
  .ncd-equivalence-diagram span {
    border-bottom: 1px solid var(--ncd-ink);
    font-size: 11px;
  }
  @keyframes ncd-jam {
    50% { transform: translateX(-4px); }
  }
  @keyframes ncd-equivalence {
    0% { opacity: 0; transform: translate(-50%, 8px) scale(0.98); }
    12%, 82% { opacity: 1; transform: translate(-50%, 0) scale(1); }
    100% { opacity: 0; transform: translate(-50%, -4px) scale(0.99); }
  }
</style>
