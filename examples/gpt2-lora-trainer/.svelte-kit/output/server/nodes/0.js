

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export const imports = ["_app/immutable/nodes/0.Q6OdBz6X.js","_app/immutable/chunks/D7odF2t9.js","_app/immutable/chunks/Cft5xwsY.js","_app/immutable/chunks/BYO0Wanp.js"];
export const stylesheets = ["_app/immutable/assets/0.3jRTbWuF.css"];
export const fonts = [];
