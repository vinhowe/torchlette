<script lang="ts">
	import { Monitor, Moon, Sun } from '@lucide/svelte';
	import { twMerge } from 'tailwind-merge';
	import { getThemeContext, type ThemeMode } from './ThemeProvider.svelte';

	type Props = {
		/** Fill the parent's height and use the app-bar's border, so it reads as a
		    built-in segment of the bar rather than a floating pill. Use inside `AppBar`. */
		integrated?: boolean;
		class?: string;
	};
	let { integrated = false, class: wrapperClass = '' }: Props = $props();

	const themeContext = getThemeContext();

	const OPTIONS = [
		{ label: 'Light', value: 'light', icon: Sun },
		{ label: 'Dark', value: 'dark', icon: Moon },
		{ label: 'System', value: 'system', icon: Monitor }
	] satisfies Array<{ label: string; value: ThemeMode; icon: typeof Sun }>;

	// A small segmented switch. Sans + sentence-case — reads as a control, not a
	// mono/uppercase command.
	// `integrated` docks it into the app bar: full height, no floating pill outline, and
	// every color (dividers, active chip, hover) reads the AppBar's `--bar-*` tokens, so
	// it rebrands together with the bar and never hardcodes a hue. The standalone pill
	// (primary-tinted border, solid-primary chip) is used on normal surfaces (gallery demo).
	const containerClasses = integrated
		? 'flex items-stretch -mt-px divide-x divide-bar-border border-l border-bar-border'
		: 'inline-flex divide-x divide-primary/20 border border-primary/30';
	// Control text (`text-base`, 12.5px, the body role) — a rung under the 13px brand
	// title, so the bar reads as title + control, not two titles. Light weight (450,
	// stem-matched to body). Icons at 12px track the label stems.
	const baseButtonClasses =
		'inline-flex items-center gap-1 px-1.25 font-sans text-base leading-none font-[450] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset disabled:pointer-events-none disabled:opacity-50 [&_svg]:shrink-0';
	const activeButtonClasses = integrated
		? 'bg-bar-accent text-bar-accent-foreground'
		: 'bg-primary text-primary-foreground';
	const inactiveButtonClasses = integrated
		? 'hover:bg-bar-accent/40 active:bg-bar-accent/70'
		: 'hover:bg-primary/(--tint-hover) active:bg-primary/(--tint-active)';

	function optionClasses(value: ThemeMode) {
		return twMerge(
			baseButtonClasses,
			integrated ? '' : 'h-control-sm',
			themeContext.theme === value ? activeButtonClasses : inactiveButtonClasses
		);
	}
</script>

<div class={twMerge(containerClasses, wrapperClass)} role="radiogroup" aria-label="Theme">
	{#each OPTIONS as option}
		{@const Icon = option.icon}
		<button
			type="button"
			role="radio"
			aria-checked={themeContext.theme === option.value}
			class={optionClasses(option.value)}
			onclick={() => themeContext.setTheme(option.value)}
		>
			<Icon size={12} strokeWidth={2} aria-hidden="true" />
			<span>{option.label}</span>
		</button>
	{/each}
</div>
