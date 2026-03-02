/**
 * Shared RuntimeEngine singleton for tests.
 *
 * Replaces the deleted src/runtime/engine-facade.ts — the facade's free
 * functions (runtimeAdd, runtimeCpu, …) only existed for test convenience.
 * Tests now use `rt.add(a, b)` instead of `runtimeAdd(a, b)`.
 */
import { RuntimeEngine } from "../../src/runtime/engine";

export const rt = new RuntimeEngine("cpu");
