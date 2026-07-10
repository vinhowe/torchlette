/**
 * Browser stub for the 'webgpu' npm package.
 *
 * The 'webgpu' package is a Node.js binding for Dawn WebGPU.
 * In browsers, we use the native WebGPU API via navigator.gpu instead.
 * This stub provides empty exports to satisfy imports but is never actually used.
 */
export const isMac = false;
export const isLinux = false;
export const isWindows = false;
export const create = () => {
  throw new Error('webgpu npm package is not available in browser. Use navigator.gpu instead.');
};
