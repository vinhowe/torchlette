/**
 * Shared WebGPU type definitions used across all webgpu/ modules.
 * These are local type aliases (not from @webgpu/types) matching the
 * original definitions in index.ts.
 */

import type { DType, BackendTensor } from "../types";

export type GPUBuffer = {
  getMappedRange(): ArrayBuffer;
  mapAsync(mode: number): Promise<void>;
  unmap(): void;
  destroy(): void;
  readonly size: number;
  readonly usage: number;
};

export type GPUComputePipeline = {
  getBindGroupLayout(index: number): unknown;
};

type GPUComputePass = {
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
  setBindGroup(index: number, group: unknown): void;
  setPipeline(pipeline: GPUComputePipeline): void;
};

export type GPUCommandEncoder = {
  beginComputePass(descriptor?: { label?: string; timestampWrites?: unknown }): GPUComputePass;
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): GPUCommandBuffer;
};

export type GPUCommandBuffer = unknown;
export type GPUBindGroup = unknown;

export type GPUQueue = {
  onSubmittedWorkDone?: () => Promise<void>;
  submit(commands: unknown[]): void;
  writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBufferView): void;
};

export type GPUDeviceLimits = {
  maxComputeWorkgroupSizeX?: number;
  maxComputeWorkgroupSizeY?: number;
  maxComputeWorkgroupsPerDimension?: number;
  maxStorageBufferBindingSize?: number;
  maxStorageBuffersPerShaderStage?: number;
  maxComputeInvocationsPerWorkgroup?: number;
  maxComputeWorkgroupStorageSize?: number;
  [key: string]: number | undefined;
};

export type GPUDevice = {
  createBindGroup(descriptor: {
    layout: unknown;
    entries: Array<{ binding: number; resource: { buffer: GPUBuffer; offset?: number; size?: number } }>;
  }): GPUBindGroup;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GPUBuffer;
  createCommandEncoder(): GPUCommandEncoder;
  createComputePipeline(descriptor: {
    layout: "auto";
    compute: { module: unknown; entryPoint: string };
  }): GPUComputePipeline;
  createShaderModule(descriptor: { code: string }): unknown;
  queue: GPUQueue;
  limits: GPUDeviceLimits;
};

export type GPUAdapterLimits = {
  maxStorageBufferBindingSize?: number;
  maxStorageBuffersPerShaderStage?: number;
};

export type GPUAdapter = {
  features?: Set<string>;
  limits?: GPUAdapterLimits;
  requestDevice(descriptor?: {
    requiredFeatures?: string[];
    requiredLimits?: GPUAdapterLimits;
  }): Promise<GPUDevice>;
};

export type WebGPUProvider = {
  requestAdapter(): Promise<GPUAdapter | null>;
};

export type WebGPUModule = {
  create: (args: string[]) => WebGPUProvider;
  globals: Record<string, unknown>;
};

export const GPUBufferUsage = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

export const GPUMapMode = {
  READ: 0x0001,
};

export const STORAGE_BUFFER_USAGE = 0x0080 | 0x0004 | 0x0008; // STORAGE | COPY_SRC | COPY_DST

export type WebGPUTensor = BackendTensor & {
  buffer: GPUBuffer;
  size: number;
  /** Strides in elements for each dimension */
  strides: number[];
  /** Offset in elements from start of buffer */
  offset: number;
  /** True if memory is contiguous (enables fast paths) */
  isContiguous: boolean;
  /** Data type of the tensor (defaults to f32 for backwards compatibility) */
  dtype: DType;
  /** True if this tensor owns the buffer (should destroy it) vs borrowing (view) */
  ownsBuffer: boolean;
  /** Destroy the GPU buffer and free memory */
  destroy(): void;
};

export type WebGPUContext = {
  provider: WebGPUProvider;
  device: GPUDevice;
  queue: GPUQueue;
  pipelines: Map<string, GPUComputePipeline>;
  /** Whether shader-f16 feature is enabled */
  f16Supported: boolean;
};

/** Narrow a generic BackendTensor to WebGPUTensor for typed property access. */
export function asGPUTensor(tensor: BackendTensor): WebGPUTensor {
  return tensor as unknown as WebGPUTensor;
}

/** Extract the GPUBuffer from a generic BackendTensor (must be a WebGPUTensor). */
export function gpuBuffer(tensor: BackendTensor): GPUBuffer {
  return asGPUTensor(tensor).buffer;
}

/** Legacy matmul constants (kept for reference, now using tiled matmul) */
export const MATMUL_WORKGROUP_X = 8;
export const MATMUL_WORKGROUP_Y = 8;
