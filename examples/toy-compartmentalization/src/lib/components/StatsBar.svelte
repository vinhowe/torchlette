<script lang="ts">
  import type { Snippet } from 'svelte';

  type Props = {
    /**
     * Stat items rendered inside the right-aligned bar. Use the `<Stat>`
     * component as the typical child:
     *   <StatsBar>
     *     <Stat label="step" value={step} />
     *     <Stat label="loss" value={lastLoss.toFixed(4)} />
     *   </StatsBar>
     * Plain text is also fine.
     */
    children: Snippet;
  };

  let { children }: Props = $props();
</script>

<div class="ml-auto flex flex-wrap items-center gap-x-[14px] gap-y-1 font-mono text-[12px] text-[rgba(0,0,0,0.54)]">
  {@render children()}
</div>
