<script lang="ts">
	import type { Snippet } from 'svelte';

	import CircleAlertIcon from '@lucide/svelte/icons/circle-alert';
	import InfoIcon from '@lucide/svelte/icons/info';
	import TriangleAlertIcon from '@lucide/svelte/icons/triangle-alert';

	import KatexBlock from '../optional/KatexBlock.svelte';
	import { twMerge } from 'tailwind-merge';

	type $$Props = {
		label: string;
		type: 'info' | 'warning' | 'error';
		children: Snippet;
	};

	let { label, type = 'info', children }: $$Props = $props();

	const borderAndBackgroundClasses = $derived(
		type === 'warning'
			? 'border-warning/60 bg-warning/15'
			: type === 'error'
				? 'border-destructive/60 bg-destructive/15'
				: 'border-border-strong/60 bg-muted/50'
	);

	const iconColorClass = $derived(
		type === 'warning' ? 'text-warning' : type === 'error' ? 'text-destructive' : 'text-muted-foreground'
	);
</script>

<div
	class={twMerge(
		'stack-tight p-1 text-foreground items-start border',
		borderAndBackgroundClasses
	)}
>
	<span
		class={twMerge(
			'flex items-center gap-1.25 type-label',
			iconColorClass
		)}
	>
		{#if type === 'info'}
			<InfoIcon class="w-3 h-3 shrink-0" strokeWidth={2.8} />
		{:else if type === 'warning'}
			<TriangleAlertIcon class="w-3 h-3 shrink-0" strokeWidth={2.8} />
		{:else if type === 'error'}
			<CircleAlertIcon class="w-3 h-3 shrink-0" strokeWidth={2.8} />
		{/if}
		<KatexBlock text={label} />
	</span>
	<span class="type-body">
		{@render children?.()}
	</span>
</div>
