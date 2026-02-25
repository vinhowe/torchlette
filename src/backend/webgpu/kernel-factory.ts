/**
 * Kernel Infrastructure Factory
 *
 * Shared plumbing for custom WebGPU compute kernels: pipeline caching,
 * uniform buffer caching, and state reset. Each kernel file calls
 * defineKernel() once at module level to get a KernelInstance, then
 * uses its methods instead of maintaining private cache Maps.
 *
 * Used by: attention-kernel, adam-kernel, layernorm-kernel,
 * cross-entropy-kernel, unscale-kernel.
 */

import type { GPUBuffer, GPUDevice, GPUComputePipeline } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";

/** A registered kernel instance with pipeline and config buffer caching. */
export interface KernelInstance {
  /** Get or create a cached compute pipeline. shaderFn is only called on cache miss. */
  getPipeline(device: GPUDevice, key: string, shaderFn: () => string): GPUComputePipeline;

  /**
   * Get or create a persistent uniform buffer, updated via writeBuffer each call.
   * The buffer is created once per unique key and reused across steps.
   */
  getConfigBuffer(device: GPUDevice, key: string, sizeBytes: number, data: ArrayBufferView): GPUBuffer;

  /** Clear all pipeline and config buffer caches. */
  reset(): void;
}

/** All registered kernel instances, for global reset. */
const allKernels: KernelInstance[] = [];

/**
 * Define a new kernel with shared pipeline and config buffer caching.
 *
 * Call once at module level:
 * ```
 * const kernel = defineKernel("crossEntropy");
 * ```
 *
 * Then in dispatch functions:
 * ```
 * const pipeline = kernel.getPipeline(device, "fwd", () => forwardShaderWGSL());
 * const config = kernel.getConfigBuffer(device, `${B}:${V}`, 8, data);
 * ```
 */
export function defineKernel(_name: string): KernelInstance {
  const pipelineCache = new Map<string, GPUComputePipeline>();
  const configCache = new Map<string, GPUBuffer>();

  const instance: KernelInstance = {
    getPipeline(device: GPUDevice, key: string, shaderFn: () => string): GPUComputePipeline {
      let pipeline = pipelineCache.get(key);
      if (!pipeline) {
        const module = device.createShaderModule({ code: shaderFn() });
        pipeline = device.createComputePipeline({
          layout: "auto",
          compute: { module, entryPoint: "main" },
        });
        pipelineCache.set(key, pipeline);
      }
      return pipeline;
    },

    getConfigBuffer(device: GPUDevice, key: string, sizeBytes: number, data: ArrayBufferView): GPUBuffer {
      let buf = configCache.get(key);
      if (!buf) {
        buf = device.createBuffer({
          size: sizeBytes,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        configCache.set(key, buf);
      }
      device.queue.writeBuffer(buf, 0, data);
      return buf;
    },

    reset(): void {
      pipelineCache.clear();
      configCache.clear();
    },
  };

  allKernels.push(instance);
  return instance;
}

/** Reset all registered kernel caches. Called from resetAllKernelCaches(). */
export function resetAllFactoryKernels(): void {
  for (const k of allKernels) {
    k.reset();
  }
}
