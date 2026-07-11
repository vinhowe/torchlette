<script lang="ts">
	import { twMerge } from 'tailwind-merge';
	import type { Snippet } from 'svelte';

	type Props = {
		/** Brand / project name, rendered at the left in sans (semibold, sentence case). */
		title?: string;
		/** Optional page/location context after the brand — dimmed, behind a hairline
		    divider (`Relay │ Settings`). Gives brand and location separate visual
		    registers instead of one long title string. */
		context?: string;
		/** Override the left/brand area entirely (takes precedence over `title`). */
		brand?: Snippet;
		/** Right-side content — actions, a ThemeToggle, etc. */
		children?: Snippet;
		class?: string;
	};

	let { title, context, brand, children, class: wrapperClass = '' }: Props = $props();
</script>

<!--
  App chrome: a thin, full-width sticky top bar. Height is a fixed integer px
  (`--bar-height`, default 24px — the macOS menu/title-bar proportion, so a 13px
  brand gets breathing room rather than filling the bar) and NOT `py-*`, so the
  full-height integrated ThemeToggle and 13px brand text center on whole pixels;
  padding-based sizing rounds unevenly and drifts the contents up/down. The bar
  chrome is driven by the `--bar-*` tokens (brand-hued purple defaults) — rebrand by pointing
  those at your hue, or override per-instance via `class`. The integrated ThemeToggle
  reads the same tokens, so the bar stays a single source of truth.
  The brand is SANS, sentence case — mono-uppercase branding is a terminal tell; mono
  stays reserved for values/code/tags.
-->
<header
	class={twMerge(
		'sticky top-0 z-30 flex h-[var(--bar-height)] shrink-0 items-center justify-between gap-8 border-t border-t-transparent border-b border-b-bar-border bg-bar pl-2 text-bar-foreground',
		wrapperClass
	)}
>
	{#if brand}
		{@render brand()}
	{:else if title}
		<span class="flex items-center gap-1.5 font-sans text-[13px] leading-none">
			<span class="font-[520]">{title}</span>
			{#if context}
				<span class="h-3 w-px bg-bar-border"></span>
				<span class="font-medium opacity-75">{context}</span>
			{/if}
		</span>
	{:else}
		<span></span>
	{/if}
	{#if children}
		<div class="flex self-stretch gap-2">{@render children()}</div>
	{/if}
</header>
