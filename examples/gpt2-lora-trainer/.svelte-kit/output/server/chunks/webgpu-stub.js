const isMac = false;
const isLinux = false;
const isWindows = false;
const create = () => {
  throw new Error("webgpu npm package is not available in browser. Use navigator.gpu instead.");
};
export {
  create,
  isLinux,
  isMac,
  isWindows
};
