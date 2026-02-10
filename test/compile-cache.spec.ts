import { beforeEach, describe, expect, it } from "vitest";

import { Engine, TraceRecorder } from "../src/engine";

describe("compiled region caching", () => {
  let engine: Engine;
  let trace: TraceRecorder;

  beforeEach(() => {
    trace = new TraceRecorder();
    engine = new Engine(trace);
    engine._debug_clearCompiledCache();
  });

  it("caches identical compiled graphs", () => {
    const compiled = engine.compile((a: number, b: number) => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t3 = engine._debug_emitLazyOp("add", {
        inputs: [t1, t2],
        shape: [2, 3],
        dtype: "f32",
      });
      return t3;
    });

    // First call - cache miss
    compiled(1, 2);
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);
    const firstStats = engine._debug_getCompiledCacheStats();
    expect(firstStats.size).toBe(1);

    // Second call with same structure - cache hit
    compiled(3, 4);
    expect(engine._debug_wasLastCompileCacheHit()).toBe(true);
    const secondStats = engine._debug_getCompiledCacheStats();
    expect(secondStats.size).toBe(1);
    expect(secondStats.entries[0].hitCount).toBe(1);
  });

  it("does not cache graphs with different shapes", () => {
    // First call with shape [2, 3]
    const compiled1 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("relu", {
        inputs: [t1],
        shape: [2, 3],
        dtype: "f32",
      });
      return t2;
    });
    compiled1();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);

    // Second call with different shape [3, 4]
    const compiled2 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [3, 4],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("relu", {
        inputs: [t1],
        shape: [3, 4],
        dtype: "f32",
      });
      return t2;
    });
    compiled2();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);

    const stats = engine._debug_getCompiledCacheStats();
    expect(stats.size).toBe(2);
  });

  it("does not cache graphs with different operations", () => {
    // First call with relu
    const compiled1 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("relu", {
        inputs: [t1],
        shape: [2, 3],
        dtype: "f32",
      });
      return t2;
    });
    compiled1();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);

    // Second call with sqrt (different op)
    const compiled2 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("sqrt", {
        inputs: [t1],
        shape: [2, 3],
        dtype: "f32",
      });
      return t2;
    });
    compiled2();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);

    const stats = engine._debug_getCompiledCacheStats();
    expect(stats.size).toBe(2);
  });

  it("does not cache graphs with different dtypes", () => {
    // First call with f32
    const compiled1 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      return t1;
    });
    compiled1();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);

    // Second call with i32
    const compiled2 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "i32",
      });
      return t1;
    });
    compiled2();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);

    const stats = engine._debug_getCompiledCacheStats();
    expect(stats.size).toBe(2);
  });

  it("caches complex graphs with multiple operations", () => {
    const compiled = engine.compile(() => {
      const a = engine._debug_emitLazyOp("input", {
        shape: [4, 4],
        dtype: "f32",
      });
      const b = engine._debug_emitLazyOp("input", {
        shape: [4, 4],
        dtype: "f32",
      });
      const c = engine._debug_emitLazyOp("matmul", {
        inputs: [a, b],
        shape: [4, 4],
        dtype: "f32",
      });
      const d = engine._debug_emitLazyOp("relu", {
        inputs: [c],
        shape: [4, 4],
        dtype: "f32",
      });
      const e = engine._debug_emitLazyOp("add", {
        inputs: [c, d],
        shape: [4, 4],
        dtype: "f32",
      });
      return e;
    });

    // First call
    compiled();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);

    // Second call - should hit cache
    compiled();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(true);

    // Third call - should hit cache again
    compiled();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(true);

    const stats = engine._debug_getCompiledCacheStats();
    expect(stats.size).toBe(1);
    expect(stats.entries[0].hitCount).toBe(2);
  });

  it("clears cache properly", () => {
    const compiled = engine.compile(() => {
      const t = engine._debug_emitLazyOp("input", { shape: [2], dtype: "f32" });
      return t;
    });

    compiled();
    expect(engine._debug_getCompiledCacheStats().size).toBe(1);

    engine._debug_clearCompiledCache();
    expect(engine._debug_getCompiledCacheStats().size).toBe(0);

    // After clear, next call should be a miss
    compiled();
    expect(engine._debug_wasLastCompileCacheHit()).toBe(false);
    expect(engine._debug_getCompiledCacheStats().size).toBe(1);
  });

  it("returns cache key for compiled graphs", () => {
    const compiled = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("add", {
        inputs: [t1, t1],
        shape: [2, 3],
        dtype: "f32",
      });
      return t2;
    });

    compiled();
    const key = engine._debug_getLastCacheKey();
    expect(key).not.toBeNull();
    expect(key?.irHash).toBeDefined();
    expect(key?.inputSignatures).toBeDefined();
    expect(key?.inputSignatures.length).toBe(2);
  });

  it("generates same cache key for structurally identical graphs", () => {
    // Create two separate engines with identical compiled functions
    const trace1 = new TraceRecorder();
    const engine1 = new Engine(trace1);

    const trace2 = new TraceRecorder();
    const engine2 = new Engine(trace2);

    const makeCompiled = (eng: Engine) =>
      eng.compile(() => {
        const a = eng._debug_emitLazyOp("input", {
          shape: [3, 3],
          dtype: "f32",
        });
        const b = eng._debug_emitLazyOp("mul", {
          inputs: [a, a],
          shape: [3, 3],
          dtype: "f32",
        });
        return b;
      });

    const compiled1 = makeCompiled(engine1);
    const compiled2 = makeCompiled(engine2);

    compiled1();
    compiled2();

    const key1 = engine1._debug_getLastCacheKey();
    const key2 = engine2._debug_getLastCacheKey();

    expect(key1).not.toBeNull();
    expect(key2).not.toBeNull();
    expect(key1?.irHash).toBe(key2?.irHash);
  });
});
