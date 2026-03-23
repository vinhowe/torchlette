/**
 * Engine-level profiler facade.
 *
 * Re-exports CPU-side profiling functions used by the engine layer.
 * The implementation lives in the WebGPU profiler (which also handles
 * GPU timestamp queries); this module provides a backend-agnostic
 * import path for engine code.
 */

export type { PlanAnalysis } from "../backend/webgpu/profiler";
export {
  getPlanAnalysisGeneration,
  getProfileModule,
  isProfilingEnabled,
  profileApiCall,
  profileOpBegin,
  profileOpEnd,
  profileSubOpBegin,
  profileSubOpEnd,
  recordFusionFallback,
  recordPlanAnalysis,
  setProfileModule,
  setProfilePhase,
} from "../backend/webgpu/profiler";
export {
  getCurrentOpLabel,
  setCurrentOpLabel,
} from "../backend/webgpu/webgpu-state";
