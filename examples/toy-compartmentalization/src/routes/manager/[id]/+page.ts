// Disable SSR for the experiment detail page. Same reason as /manager: the
// singleton ExperimentClient opens a WebSocket at module scope, which isn't
// available during server-side rendering.
export const ssr = false;

// And don't prerender — the [id] segment is dynamic and we have no way to
// enumerate ids at build time. The fallback: 'index.html' config in
// svelte.config.js lets this route SPA-route correctly.
export const prerender = false;
