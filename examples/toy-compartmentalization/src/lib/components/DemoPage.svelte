<script lang="ts">
  import type { Snippet } from 'svelte';
  import { ActionButton } from 'piston-controls';
  import type { DemoRoute } from './types';

  type Props = {
    /** Page title — large bold heading. */
    title: string;
    /** Lead paragraph(s) — markdown-ish text describing the demo. */
    lead?: Snippet;
    /** Which entry in the subnav is the current page. */
    currentRoute: DemoRoute;

    /** Optional GPU error to display as a banner above the controls. */
    gpuError?: string;

    /** Whether GPU is ready (disables train/reset until true). */
    gpuReady: boolean;
    /** Whether training is currently running (disables train, enables stop). */
    training: boolean;

    onTrain: () => void | Promise<void>;
    onStop: () => void;
    onReset: () => void | Promise<void>;
    onDefaults: () => void | Promise<void>;
    onClear: () => void | Promise<void>;

    /** Optional slot: explanatory content rendered between the header and the controls. */
    intro?: Snippet;
    /** Slot: control groups (BorderedGroups) — typically a 3-col grid. */
    controls: Snippet;
    /** Slot: stats line shown to the right of the action buttons. */
    stats?: Snippet;
    /** Slot: figures (sections with charts/visualizations). */
    figures: Snippet;
  };

  let {
    title,
    lead,
    currentRoute,
    gpuError,
    gpuReady,
    training,
    onTrain,
    onStop,
    onReset,
    onDefaults,
    onClear,
    intro,
    controls,
    stats,
    figures,
  }: Props = $props();

  const ROUTES: { route: DemoRoute; href: string; label: string }[] = [
    { route: 'mess3',    href: '/',         label: 'mess3' },
    { route: 'bio',      href: '/bio',      label: 'bio' },
    { route: 'xor',      href: '/xor',      label: 'xor' },
    { route: 'brackets', href: '/brackets', label: 'brackets' },
    { route: 'rnn',      href: '/rnn',      label: 'rnn' },
  ];
</script>

<div class="mx-auto max-w-[860px] px-6 pt-12 pb-24 text-[rgba(0,0,0,0.84)]">
  <nav class="mb-8 flex gap-[18px] font-mono text-[12px]">
    {#each ROUTES as { route, href, label }}
      {@const current = route === currentRoute}
      <a
        {href}
        class="uppercase tracking-[0.06em] no-underline {current
          ? 'text-[rgba(0,0,0,0.84)] font-semibold'
          : 'text-[rgba(0,0,0,0.54)] hover:text-[rgba(0,0,0,0.84)]'}"
      >
        {label}
      </a>
    {/each}
  </nav>

  <header class="mb-9">
    <h1 class="mb-3 text-[32px] font-bold leading-[1.15] tracking-[-0.02em]">{title}</h1>
    {#if lead}
      <div class="max-w-[680px] text-[16px] leading-[1.65] text-[rgba(0,0,0,0.7)]">
        {@render lead()}
      </div>
    {/if}
  </header>

  {#if gpuError}
    <div class="mb-5 border-l-[3px] border-l-[#d62728] bg-[rgba(214,39,40,0.05)] px-[14px] py-2 text-[13px]">
      WebGPU: {gpuError}
    </div>
  {/if}

  {#if intro}
    <div class="mb-6">
      {@render intro()}
    </div>
  {/if}

  <div class="mb-4 grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-[12px]">
    {@render controls()}
  </div>

  <div class="mb-9 flex flex-wrap items-center gap-x-[12px] gap-y-2 border-y border-[rgba(0,0,0,0.08)] py-3">
    <ActionButton color="blue" onclick={onTrain} disabled={!gpuReady || training}>train</ActionButton>
    <ActionButton color="gray" onclick={onStop} disabled={!training}>stop</ActionButton>
    <ActionButton color="gray" onclick={onReset} disabled={!gpuReady}>reset</ActionButton>
    <ActionButton color="gray" onclick={onDefaults}>defaults</ActionButton>
    <ActionButton color="gray" onclick={onClear}>clear</ActionButton>

    {#if stats}{@render stats()}{/if}
  </div>

  <div class="flex flex-col gap-11">
    {@render figures()}
  </div>
</div>
