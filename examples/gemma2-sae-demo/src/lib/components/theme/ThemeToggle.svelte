<script lang="ts">
	import { Monitor, Moon, Sun } from '@lucide/svelte';
	import { twMerge } from 'tailwind-merge';
	import { getThemeContext, type ThemeMode } from './ThemeProvider.svelte';

	const themeContext = getThemeContext();

	const OPTIONS = [
		{ label: 'Light', value: 'light', icon: Sun },
		{ label: 'Dark', value: 'dark', icon: Moon },
		{ label: 'System', value: 'system', icon: Monitor }
	] satisfies Array<{ label: string; value: ThemeMode; icon: typeof Sun }>;

	const baseButtonClasses =
		'inline-flex items-center gap-1 border-r border-border px-1.5 py-0.5 transition-colors last:border-r-0 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 type-button';
	const activeButtonClasses = 'bg-foreground text-background';
	const inactiveButtonClasses =
		'bg-card text-muted-foreground hover:bg-muted hover:text-foreground dark:bg-card dark:hover:bg-muted';

	function optionClasses(value: ThemeMode) {
		return twMerge(
			baseButtonClasses,
			themeContext.theme === value ? activeButtonClasses : inactiveButtonClasses
		);
	}
</script>

<div
	class="inline-flex border border-border bg-card"
	role="radiogroup"
	aria-label="Theme"
>
	{#each OPTIONS as option}
		{@const Icon = option.icon}
		<button
			type="button"
			role="radio"
			aria-checked={themeContext.theme === option.value}
			class={optionClasses(option.value)}
			onclick={() => themeContext.setTheme(option.value)}
		>
			<Icon size={13} strokeWidth={2} aria-hidden="true" />
			<span>{option.label}</span>
		</button>
	{/each}
</div>
