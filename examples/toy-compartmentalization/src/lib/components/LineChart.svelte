<script lang="ts">
  import { onMount } from 'svelte';

  type Props = {
    /**
     * Echarts option object. The chart re-runs `setOption(option, true)`
     * whenever this changes (use `$derived(...)` upstream so reactivity
     * triggers correctly).
     */
    option: unknown;
    /** Container height in px (default 220 — figures.small uses 160). */
    height?: number;
  };

  let { option, height = 220 }: Props = $props();

  let el: HTMLDivElement | undefined = $state();
  let chart: { setOption: (opt: unknown, notMerge?: boolean) => void; resize: () => void; dispose: () => void } | null = null;
  let ready = $state(false);

  onMount(() => {
    let disposed = false;
    let resizeObs: ResizeObserver | null = null;
    (async () => {
      const echarts = await import('echarts');
      if (disposed || !el) return;
      chart = echarts.init(el);
      ready = true;
      if (typeof ResizeObserver !== 'undefined') {
        resizeObs = new ResizeObserver(() => chart?.resize());
        resizeObs.observe(el);
      }
    })();
    return () => {
      disposed = true;
      resizeObs?.disconnect();
      chart?.dispose();
      chart = null;
    };
  });

  $effect(() => {
    if (ready && chart && option) {
      chart.setOption(option, true);
    }
  });
</script>

<div bind:this={el} class="w-full" style="height: {height}px"></div>
