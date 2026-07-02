<script lang="ts">
	import type { Snippet } from 'svelte';

	import KatexBlock from '../optional/KatexBlock.svelte';
	import Citations, { type CitationEntries as CitationsType } from '../feedback/Citations.svelte';
	import { twMerge } from 'tailwind-merge';

	type $$Props = {
		title?: string;
		id?: string;
		citations?: CitationsType;
		class?: string;
		headerClass?: string;
		contentClass?: string | null;
		header?: Snippet;
		action?: Snippet;
		children?: Snippet;
	};

	let {
		title,
		id,
		citations,
		class: rootClass = '',
		headerClass = 'bg-panel border-b border-border px-1 py-0.5 flex justify-between items-center text-panel-foreground gap-2',
		contentClass = 'p-1.5 stack-field',
		header,
		action,
		children
	}: $$Props = $props();

	// svelte-ignore state_referenced_locally -- one-time construction-time validation of the initial props
	if (!title && !header) {
		throw new Error('Either title or header must be provided');
	}
</script>

<div {id} class={twMerge(`border border-border overflow-hidden bg-card`, rootClass)}>
	<div class={headerClass}>
		{#if title}
			<div class="flex flex-col">
				<h3 class="type-title">
					<KatexBlock text={title} />
				</h3>
				{#if citations}
					<Citations {citations} />
				{/if}
			</div>
		{:else}
			{@render header?.()}
		{/if}
		{#if action}
			{@render action?.()}
		{/if}
	</div>
	{#if contentClass}
		<div class={contentClass}>
			{@render children?.()}
		</div>
	{:else}
		{@render children?.()}
	{/if}
</div>
