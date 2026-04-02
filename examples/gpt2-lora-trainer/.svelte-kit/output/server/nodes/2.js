

export const index = 2;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_page.svelte.js')).default;
export const universal = {
  "ssr": false
};
export const universal_id = "src/routes/+page.ts";
export const imports = ["_app/immutable/nodes/2.BC7jBwEy.js","_app/immutable/chunks/BFmezKYF.js","_app/immutable/chunks/DtExVYyI.js","_app/immutable/chunks/Cft5xwsY.js","_app/immutable/chunks/BYO0Wanp.js","_app/immutable/chunks/D7odF2t9.js","_app/immutable/chunks/C679vkDn.js","_app/immutable/chunks/DH20m1UJ.js"];
export const stylesheets = [];
export const fonts = [];
