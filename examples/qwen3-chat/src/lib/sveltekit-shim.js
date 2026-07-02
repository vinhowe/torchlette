// Shim for SvelteKit's `$app/environment` — this app is plain Vite+Svelte,
// and everything here only runs in the browser.
export const browser = true;
export const dev = import.meta.env.DEV;
export const building = false;
