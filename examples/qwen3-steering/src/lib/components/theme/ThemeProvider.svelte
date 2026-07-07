<script module lang="ts">
	import { getContext, setContext } from 'svelte';

	export type ThemeMode = 'light' | 'dark' | 'system';
	export type ResolvedTheme = 'light' | 'dark';

	export interface ThemeContext {
		readonly theme: ThemeMode;
		readonly resolvedTheme: ResolvedTheme;
		setTheme: (theme: ThemeMode) => void;
	}

	const THEME_CONTEXT_KEY = Symbol('sequence-ui-theme');

	export function getThemeContext() {
		const context = getContext<ThemeContext>(THEME_CONTEXT_KEY);

		if (!context) {
			throw new Error('ThemeToggle must be used inside ThemeProvider.');
		}

		return context;
	}
</script>

<script lang="ts">
	import { browser } from '$app/environment';
	import type { Snippet } from 'svelte';

	interface Props {
		children?: Snippet;
		defaultTheme?: ThemeMode;
	}

	const STORAGE_KEY = 'sequence-ui-theme';

	let { children, defaultTheme = 'system' }: Props = $props();
	// svelte-ignore state_referenced_locally -- defaultTheme is intentionally used only as the initial seed
	let theme = $state<ThemeMode>(defaultTheme);
	let resolvedTheme = $state<ResolvedTheme>('light');

	function isThemeMode(value: string | null): value is ThemeMode {
		return value === 'light' || value === 'dark' || value === 'system';
	}

	function systemTheme(): ResolvedTheme {
		return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
	}

	function resolveTheme(mode: ThemeMode): ResolvedTheme {
		return mode === 'system' ? systemTheme() : mode;
	}

	function applyTheme(mode: ThemeMode) {
		if (!browser) return;

		resolvedTheme = resolveTheme(mode);
		document.documentElement.classList.toggle('dark', resolvedTheme === 'dark');
	}

	function setTheme(nextTheme: ThemeMode) {
		theme = nextTheme;

		if (browser) {
			localStorage.setItem(STORAGE_KEY, nextTheme);
		}

		applyTheme(nextTheme);
	}

	setContext<ThemeContext>(THEME_CONTEXT_KEY, {
		get theme() {
			return theme;
		},
		get resolvedTheme() {
			return resolvedTheme;
		},
		setTheme
	});

	$effect(() => {
		if (!browser) return;

		const storedTheme = localStorage.getItem(STORAGE_KEY);
		theme = isThemeMode(storedTheme) ? storedTheme : defaultTheme;
		applyTheme(theme);

		const media = window.matchMedia('(prefers-color-scheme: dark)');
		const handleSystemChange = () => {
			if (theme === 'system') {
				applyTheme('system');
			}
		};

		media.addEventListener('change', handleSystemChange);

		return () => {
			media.removeEventListener('change', handleSystemChange);
		};
	});
</script>

{@render children?.()}
