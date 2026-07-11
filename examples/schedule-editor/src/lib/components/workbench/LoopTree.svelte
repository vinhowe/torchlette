<script lang="ts">
  import type { LoopNode } from "../../schedule-state";
  import LoopTree from "./LoopTree.svelte";

  let { loops, depth = 0 }: { loops: LoopNode[]; depth?: number } = $props();
</script>

<div class="stack-tight">
  {#each loops as loop}
    <div class="border border-border bg-card">
      <div class="grid grid-cols-[1fr_auto] items-center gap-2 pad-box" style={`padding-left:${4 + depth * 8}px`}>
        <span class="type-label">{loop.axis}</span>
        <span class="type-tag">{loop.execution}</span>
        <span class="type-code text-muted-foreground">{loop.extent}</span>
        <span class="type-value">{loop.id}</span>
      </div>
      {#if loop.children?.length}
        <div class="border-t border-border pad-box">
          <LoopTree loops={loop.children} depth={depth + 1} />
        </div>
      {/if}
    </div>
  {/each}
</div>
