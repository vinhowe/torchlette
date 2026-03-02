import { describe, expect, it } from "vitest";
import {
  allocateOutputBuffer,
  type BufferArena,
  clearActiveArena,
  destroyArena,
  getArenaResolveIndex,
  getWebGPUDevice,
  initWebGPU,
  isArenaBuffer,
  resolveOutputBuffer,
  setActiveArena,
  setArenaResolveIndexTo,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

function requireDevice() {
  const ctx = getWebGPUDevice();
  if (!ctx) throw new Error("WebGPU device not available");
  return ctx;
}

function makeArena(resolveSlots: number, allocSlots: number): BufferArena {
  return {
    resolve: new Array(resolveSlots).fill(undefined),
    alloc: new Array(allocSlots).fill(undefined),
  };
}

describe.skipIf(cpuOnly)("buffer arena", { timeout: 30000 }, () => {
  it("allocates buffer from arena and marks it as arena buffer", async () => {
    await initWebGPU();
    const device = requireDevice();
    const arena = makeArena(5, 5);

    setActiveArena(arena);
    const buf = resolveOutputBuffer(device, 4096, []);
    expect(buf).toBeDefined();
    expect(isArenaBuffer(buf)).toBe(true);
    clearActiveArena();

    destroyArena(arena);
  });

  it("reuses same buffer across multiple activations", async () => {
    await initWebGPU();
    const device = requireDevice();
    const arena = makeArena(5, 5);

    // First activation: allocate buffer at slot 0
    setActiveArena(arena);
    const buf1 = resolveOutputBuffer(device, 4096, []);
    clearActiveArena();

    // Second activation: should reuse the same buffer
    setActiveArena(arena);
    const buf2 = resolveOutputBuffer(device, 4096, []);
    clearActiveArena();

    expect(buf2).toBe(buf1); // Exact same GPUBuffer object

    destroyArena(arena);
  });

  it("tracks resolve index correctly", async () => {
    await initWebGPU();
    const device = requireDevice();
    const arena = makeArena(5, 5);

    setActiveArena(arena);
    expect(getArenaResolveIndex()).toBe(0);

    resolveOutputBuffer(device, 1024, []);
    expect(getArenaResolveIndex()).toBe(1);

    resolveOutputBuffer(device, 2048, []);
    expect(getArenaResolveIndex()).toBe(2);

    // Can reset index
    setArenaResolveIndexTo(0);
    expect(getArenaResolveIndex()).toBe(0);

    clearActiveArena();
    destroyArena(arena);
  });

  it("multiple arenas have independent buffer sets", async () => {
    await initWebGPU();
    const device = requireDevice();
    const arena1 = makeArena(3, 3);
    const arena2 = makeArena(3, 3);

    // Allocate from arena1
    setActiveArena(arena1);
    const buf1a = resolveOutputBuffer(device, 4096, []);
    const buf1b = resolveOutputBuffer(device, 8192, []);
    clearActiveArena();

    // Allocate from arena2
    setActiveArena(arena2);
    const buf2a = resolveOutputBuffer(device, 4096, []);
    const buf2b = resolveOutputBuffer(device, 8192, []);
    clearActiveArena();

    // Different arenas should produce different buffers
    expect(buf1a).not.toBe(buf2a);
    expect(buf1b).not.toBe(buf2b);

    // Both should be arena buffers
    expect(isArenaBuffer(buf1a)).toBe(true);
    expect(isArenaBuffer(buf2a)).toBe(true);

    destroyArena(arena1);
    destroyArena(arena2);
  });

  it("allocateOutputBuffer uses separate alloc index from resolveOutputBuffer", async () => {
    await initWebGPU();
    const device = requireDevice();
    const arena = makeArena(5, 5);

    setActiveArena(arena);
    const resolve1 = resolveOutputBuffer(device, 4096, []);
    const alloc1 = allocateOutputBuffer(2048);
    const resolve2 = resolveOutputBuffer(device, 4096, []);
    clearActiveArena();

    // All three should be different buffers
    expect(resolve1).not.toBe(alloc1);
    expect(resolve1).not.toBe(resolve2);
    expect(alloc1).not.toBe(resolve2);

    // All should be arena buffers
    expect(isArenaBuffer(resolve1)).toBe(true);
    expect(isArenaBuffer(alloc1)).toBe(true);
    expect(isArenaBuffer(resolve2)).toBe(true);

    destroyArena(arena);
  });

  it("destroyArena removes buffers from arena buffer set", async () => {
    await initWebGPU();
    const device = requireDevice();
    const arena = makeArena(3, 3);

    setActiveArena(arena);
    const buf = resolveOutputBuffer(device, 4096, []);
    clearActiveArena();

    expect(isArenaBuffer(buf)).toBe(true);

    destroyArena(arena);

    expect(isArenaBuffer(buf)).toBe(false);
  });
});
