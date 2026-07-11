<script lang="ts">
import { axisById } from "../../ncd/model";
import type { NcdLevel, NcdTerm, PartitionKind } from "../../ncd/types";

type Props = {
  term: NcdTerm;
  onPartitionDrop: (axisId: string, kind: PartitionKind) => void;
  onResidencyDrop: (wireId: string, column: number, level: NcdLevel) => void;
};

let { term, onPartitionDrop, onResidencyDrop }: Props = $props();

function partition(axisId: string) {
  return term.decorations.partitions.find((item) => item.axisId === axisId);
}

function divisibility(axisId: string) {
  return term.decorations.divisibility.find((item) => item.axisId === axisId);
}

function residency(wireId: string, column: number) {
  return term.decorations.residency.find(
    (item) => item.wireId === wireId && item.column === column,
  );
}

function boxAt(column: number) {
  return term.semantic.boxes.find((box) => box.column === column);
}

function parsePayload(
  event: DragEvent,
):
  | { type: "partition"; kind: PartitionKind }
  | { type: "level"; level: NcdLevel }
  | null {
  try {
    return JSON.parse(
      event.dataTransfer?.getData("application/x-torchlette-ncd") ?? "",
    ) as
      | { type: "partition"; kind: PartitionKind }
      | { type: "level"; level: NcdLevel };
  } catch {
    return null;
  }
}

function dropAxis(event: DragEvent, axisId: string): void {
  event.preventDefault();
  const payload = parsePayload(event);
  if (payload?.type === "partition") onPartitionDrop(axisId, payload.kind);
}

function dropResidency(event: DragEvent, wireId: string, column: number): void {
  event.preventDefault();
  const payload = parsePayload(event);
  if (payload?.type === "level") onResidencyDrop(wireId, column, payload.level);
}
</script>

<div class="overflow-x-auto border border-border bg-background pad-box" aria-label="Neural circuit diagram">
  <div
    class="grid min-w-[62rem] gap-1"
    style={`grid-template-columns: 5rem repeat(${term.semantic.columns.length}, minmax(7rem, 1fr))`}
  >
    <div class="type-tag text-muted-foreground">ARRAY</div>
    {#each term.semantic.columns as column}
      <div class="border-b border-border pb-1 text-center type-tag text-muted-foreground">
        {column.index} · {column.label}
      </div>
    {/each}

    <div class="type-tag text-muted-foreground">FUNCTION</div>
    {#each term.semantic.columns as column}
      {@const box = boxAt(column.index)}
      <div class="flex min-h-12 items-center justify-center">
        {#if box}
          <div class={box.streamability.kind === "decomposed" ? "w-full border border-primary bg-primary/10 pad-box text-center" : "w-full border border-warning bg-warning/10 pad-box text-center"}>
            <div class="type-label">{box.label}</div>
            <div class={box.streamability.kind === "decomposed" ? "type-tag text-success" : "type-tag text-warning-strong"}>
              {box.streamability.kind === "decomposed" ? "HEAD / BODY" : "NOT STREAMABLE"}
            </div>
          </div>
        {/if}
      </div>
    {/each}

    {#each term.semantic.wires as wire}
      <div class={`flex flex-col justify-center border-r border-border pr-1 ${wire.tupleGroup ? "border-y border-dashed border-border-strong" : ""}`}>
        <span class="type-label">{wire.label}</span>
        <span class="type-tag">{wire.axisIds.join(" · ")}</span>
      </div>
      {#each term.semantic.columns as column}
        {@const state = residency(wire.id, column.index)}
        <div class={`min-h-16 border ${wire.tupleGroup ? "border-dashed" : "border-solid"} border-border pad-box`}>
          {#if state}
            <div
              class={state.level === "l0" ? "stack-tight text-foreground" : "stack-tight text-primary-accent"}
              role="button"
              tabindex="0"
              aria-label={`${wire.label} column ${column.index} residency ${state.level}`}
              ondragover={(event) => event.preventDefault()}
              ondrop={(event) => dropResidency(event, wire.id, column.index)}
            >
              <div class="flex items-center justify-between gap-1">
                <span class="type-tag">{state.level}</span>
                <span class="type-value">{wire.elementBytes}B</span>
              </div>
              {#each wire.axisIds as axisId}
                {@const axis = axisById(term, axisId)}
                {@const part = partition(axisId)}
                {@const divisor = divisibility(axisId)}
                <div
                  class="flex h-control-sm items-center gap-1 border border-current px-1"
                  role="button"
                  tabindex="0"
                  aria-label={`${wire.label} axis ${axis.label}`}
                  ondragover={(event) => event.preventDefault()}
                  ondrop={(event) => dropAxis(event, axisId)}
                  title="Drop a group or stream label here"
                >
                  <span class="type-code">{axis.label}</span>
                  <span class="h-px min-w-2 flex-1 bg-current"></span>
                  <span class="type-value">{axis.size}</span>
                  {#if divisor}<sup class="type-tag">|{divisor.multiple}</sup>{/if}
                  {#if part}<span class="border border-current px-0.5 type-tag">{part.label}={part.size}</span>{/if}
                </div>
              {/each}
            </div>
          {:else}
            <div class="flex h-full min-h-12 items-center justify-center type-tag text-subtle-foreground">·</div>
          {/if}
        </div>
      {/each}
    {/each}
  </div>
</div>
