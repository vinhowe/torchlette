// Fully client-side app (WebGPU + HF OAuth live in the browser). No SSR.
// SPA mode: adapter-static emits a fallback shell (svelte.config.js) that
// hydrates on the client.
export const ssr = false;
