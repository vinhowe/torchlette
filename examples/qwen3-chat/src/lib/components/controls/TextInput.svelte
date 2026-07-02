<script lang="ts">
	import FormLabel from './FormLabel.svelte';
	import ResetValueButton from './ResetValueButton.svelte';

	type $$Props = {
		label?: string;
		value: string | number;
		id: string;
		class?: string;
		placeholder?: string;
		type?: string;
		hasDefaultValue?: boolean;
		onReset?: () => void;
	};

	let {
		label,
		value = $bindable(),
		id,
		class: wrapperClass = '',
		placeholder = '',
		type = 'text',
		hasDefaultValue = false,
		onReset = undefined
	}: $$Props = $props();
</script>

<div class="relative stack-tight {wrapperClass}">
	{#if label}
		<FormLabel forInputId={id} value={label} />
	{/if}
	<div class="flex gap-1.5 items-center">
		<input
			{id}
			{type}
			{placeholder}
			bind:value
			class="block w-full border border-border bg-card py-0.5 px-1 text-foreground placeholder:text-subtle-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring type-body"		/>
		{#if onReset}
			<ResetValueButton {hasDefaultValue} {onReset} />
		{/if}
	</div>
</div>
