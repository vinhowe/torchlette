<script lang="ts">
	import { twMerge } from 'tailwind-merge';
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';

	type ActionButtonColor = 'blue' | 'red' | 'green' | 'yellow' | 'purple' | 'gray';

	interface Props extends HTMLButtonAttributes {
		color?: ActionButtonColor;
		colorClass?: string;
		children?: Snippet;
		highlighted?: boolean;
	}

	let {
		color,
		colorClass,
		class: additionalClasses,
		highlighted = false,
		disabled,
		children,
		...restProps
	}: Props = $props();

	const COLOR_MAP: Record<ActionButtonColor, string> = {
		blue: 'text-blue-900 bg-gradient-to-t from-blue-300 to-blue-100 border-blue-600 dark:text-blue-100 dark:from-blue-800 dark:to-blue-700 dark:border-blue-500',
		red: 'text-red-900 bg-gradient-to-t from-red-300 to-red-100 border-red-600 dark:text-red-100 dark:from-red-800 dark:to-red-700 dark:border-red-500',
		green:
			'text-green-900 bg-gradient-to-t from-green-300 to-green-100 border-green-600 dark:text-green-100 dark:from-green-800 dark:to-green-700 dark:border-green-500',
		yellow:
			'text-yellow-900 bg-gradient-to-t from-yellow-300 to-yellow-100 border-yellow-600 dark:text-yellow-100 dark:from-yellow-800 dark:to-yellow-700 dark:border-yellow-500',
		purple:
			'text-purple-900 bg-gradient-to-t from-purple-300 to-purple-100 border-purple-600 dark:text-purple-100 dark:from-purple-800 dark:to-purple-700 dark:border-purple-500',
		gray: 'text-gray-900 bg-gradient-to-t from-gray-300 to-gray-100 border-gray-600 dark:text-gray-100 dark:from-gray-800 dark:to-gray-700 dark:border-gray-500'
	};

	const HIGHLIGHT_COLOR_MAP: Record<ActionButtonColor, string> = {
		blue: 'text-blue-100 bg-gradient-to-t from-blue-700 to-blue-500 border-blue-800 animate-pulse dark:text-blue-950 dark:from-blue-500 dark:to-blue-400 dark:border-blue-300',
		red: 'text-red-100 bg-gradient-to-t from-red-700 to-red-500 border-red-800 animate-pulse dark:text-red-950 dark:from-red-500 dark:to-red-400 dark:border-red-300',
		green:
			'text-green-100 bg-gradient-to-t from-green-700 to-green-500 border-green-800 animate-pulse dark:text-green-950 dark:from-green-500 dark:to-green-400 dark:border-green-300',
		yellow:
			'text-yellow-950 bg-gradient-to-t from-yellow-400 to-yellow-300 border-yellow-600 animate-pulse dark:text-yellow-950 dark:from-yellow-300 dark:to-yellow-200 dark:border-yellow-200',
		purple:
			'text-purple-100 bg-gradient-to-t from-purple-700 to-purple-500 border-purple-800 animate-pulse dark:text-purple-950 dark:from-purple-500 dark:to-purple-400 dark:border-purple-300',
		gray: 'text-gray-100 bg-gradient-to-t from-gray-700 to-gray-500 border-gray-800 animate-pulse dark:text-gray-950 dark:from-gray-400 dark:to-gray-300 dark:border-gray-200'
	};

	const baseClasses =
		'inline-flex h-7.5 cursor-pointer items-center justify-center border px-2 py-1 transition-[filter,opacity] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 disabled:saturate-50 type-button';

	const computedColorClass = $derived(
		color
			? highlighted
				? HIGHLIGHT_COLOR_MAP[color]
				: COLOR_MAP[color]
			: colorClass || COLOR_MAP.blue
	);

	const normalizedAdditionalClasses = $derived(
		typeof additionalClasses === 'string' ? additionalClasses : ''
	);
</script>

<button
	type="button"
	class={twMerge(baseClasses, computedColorClass, normalizedAdditionalClasses)}
	{disabled}
	{...restProps}
>
	{@render children?.()}
</button>
