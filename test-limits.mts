import { initWebGPU } from "./src/backend/webgpu/index.js";
const r = await initWebGPU();
console.log("Result:", r);
