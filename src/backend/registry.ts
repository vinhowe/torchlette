import { cpuBackend } from "./cpu";
import { mockBackend } from "./mock";
import type { Backend, BackendOps } from "./types";

const backends = new Map<string, Backend>();
backends.set(cpuBackend.name, cpuBackend);
backends.set(mockBackend.name, mockBackend);

let activeBackend = cpuBackend;

export function getBackend(name: string): Backend | undefined {
  return backends.get(name);
}

export function registerBackend(backend: Backend): void {
  backends.set(backend.name, backend);
}

export function setBackend(name: string): Backend {
  const backend = backends.get(name);
  if (!backend) {
    throw new Error(`Unknown backend: ${name}`);
  }
  activeBackend = backend;
  return backend;
}

export function getActiveBackend(): Backend {
  return activeBackend;
}

export function ops(): BackendOps {
  return activeBackend.ops;
}

export function withBackend<T>(name: string, fn: (backend: Backend) => T): T {
  const previous = activeBackend;
  const backend = setBackend(name);
  try {
    return fn(backend);
  } finally {
    activeBackend = previous;
  }
}
