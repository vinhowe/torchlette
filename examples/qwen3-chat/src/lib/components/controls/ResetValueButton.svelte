<script lang="ts">
	import { UndoIcon } from '@lucide/svelte/icons';

	let {
		hasDefaultValue = false,
		onReset
	}: {
		hasDefaultValue: boolean;
		onReset: () => void;
	} = $props();
</script>

<button
	type="button"
	class="w-3.75 h-3.75 text-muted-foreground disabled:text-border enabled:cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
	title="Reset to default"
	disabled={hasDefaultValue}
	onclick={() => {
		onReset?.();
	}}
>
	<UndoIcon class="w-3.5 h-3.5" strokeWidth={2.75} />
</button>
