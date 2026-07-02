<script lang="ts">
	import type { Snippet } from 'svelte';

	import KatexBlock from '../optional/KatexBlock.svelte';

	type $$Props = {
		label: string;
		children: Snippet;
		funFact?: Snippet;
	};

	let { label, children, funFact }: $$Props = $props();
</script>

<div
	class="stack-tight p-1 bg-muted/70 border border-border-strong/60 text-foreground"
>
	<span class="text-subtle-foreground type-label"
		><KatexBlock text={label} /></span
	>
	<span class="flex items-baseline gap-x-1">
		<span class="type-value">
			{@render children?.()}
		</span>
		{#if funFact}
			<span class="text-right w-full text-muted-foreground type-caption">
				{@render funFact?.()}
			</span>
		{/if}
	</span>
</div>
