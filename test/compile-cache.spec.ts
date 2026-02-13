import { beforeEach, describe, expect, it } from "vitest";

import { Engine, TraceRecorder } from "../src/engine";
import { hashIRGraph } from "../src/engine/compile-cache";
import type { IRGraph, IRNode } from "../src/engine/ir";

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

  it("different scalar values produce different cache keys", () => {
    // Two compile calls with identical structure but different scalar constants
    const compiled1 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("mul", {
        inputs: [t1],
        shape: [2, 3],
        dtype: "f32",
        scalarValues: [2.0],
      });
      return t2;
    });

    const compiled2 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("mul", {
        inputs: [t1],
        shape: [2, 3],
        dtype: "f32",
        scalarValues: [3.0],
      });
      return t2;
    });

    compiled1();
    const key1 = engine._debug_getLastCacheKey();

    compiled2();
    const key2 = engine._debug_getLastCacheKey();

    expect(key1).not.toBeNull();
    expect(key2).not.toBeNull();
    // Different scalar values → different irHash
    expect(key1?.irHash).not.toBe(key2?.irHash);
  });

  it("same scalar values produce same cache keys", () => {
    const compiled1 = engine.compile(() => {
      const t1 = engine._debug_emitLazyOp("input", {
        shape: [2, 3],
        dtype: "f32",
      });
      const t2 = engine._debug_emitLazyOp("mul", {
        inputs: [t1],
        shape: [2, 3],
        dtype: "f32",
        scalarValues: [2.0],
      });
      return t2;
    });

    // First call
    compiled1();
    const key1 = engine._debug_getLastCacheKey();

    // Second call with same scalar
    compiled1();
    const key2 = engine._debug_getLastCacheKey();

    expect(key1).not.toBeNull();
    expect(key2).not.toBeNull();
    expect(key1?.irHash).toBe(key2?.irHash);
    expect(engine._debug_wasLastCompileCacheHit()).toBe(true);
  });
});

describe("scalar canonicalization in hashIRGraph (§8.2.1)", () => {
  function makeGraph(nodes: IRNode[]): IRGraph {
    return { epoch: 1, nodes, fusionGroups: [] };
  }

  it("different scalar values produce different hashes", () => {
    const graph1 = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [2, 3], dtype: "f32" },
      { id: 1, op: "mul", epoch: 1, kind: "lazy_op", inputs: [0], shape: [2, 3], dtype: "f32", scalarValues: [2.0] },
    ]);

    const graph2 = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [2, 3], dtype: "f32" },
      { id: 1, op: "mul", epoch: 1, kind: "lazy_op", inputs: [0], shape: [2, 3], dtype: "f32", scalarValues: [3.0] },
    ]);

    expect(hashIRGraph(graph1)).not.toBe(hashIRGraph(graph2));
  });

  it("+0 vs -0 produces different hashes", () => {
    const graphPos = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" },
      { id: 1, op: "add", epoch: 1, kind: "lazy_op", inputs: [0], shape: [4], dtype: "f32", scalarValues: [+0.0] },
    ]);

    const graphNeg = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [4], dtype: "f32" },
      { id: 1, op: "add", epoch: 1, kind: "lazy_op", inputs: [0], shape: [4], dtype: "f32", scalarValues: [-0.0] },
    ]);

    expect(hashIRGraph(graphPos)).not.toBe(hashIRGraph(graphNeg));
  });

  it("NaN canonicalization: different NaN payloads produce same hash", () => {
    // Create different NaN values via typed array manipulation
    const buf1 = new ArrayBuffer(8);
    const view1 = new DataView(buf1);
    view1.setUint32(0, 0x7ff80001, false); // quiet NaN with payload 1
    const nan1 = view1.getFloat64(0, false);

    const buf2 = new ArrayBuffer(8);
    const view2 = new DataView(buf2);
    view2.setUint32(0, 0x7ff80002, false); // quiet NaN with payload 2
    const nan2 = view2.getFloat64(0, false);

    // Both should be NaN but with different bit patterns
    expect(Number.isNaN(nan1)).toBe(true);
    expect(Number.isNaN(nan2)).toBe(true);

    const graphNaN1 = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [2], dtype: "f32" },
      { id: 1, op: "mul", epoch: 1, kind: "lazy_op", inputs: [0], shape: [2], dtype: "f32", scalarValues: [nan1] },
    ]);

    const graphNaN2 = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [2], dtype: "f32" },
      { id: 1, op: "mul", epoch: 1, kind: "lazy_op", inputs: [0], shape: [2], dtype: "f32", scalarValues: [nan2] },
    ]);

    // Canonical NaN → same hash
    expect(hashIRGraph(graphNaN1)).toBe(hashIRGraph(graphNaN2));
  });

  it("no scalars produces same hash as before (backward compatible)", () => {
    const graph = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [2, 3], dtype: "f32" },
      { id: 1, op: "relu", epoch: 1, kind: "lazy_op", inputs: [0], shape: [2, 3], dtype: "f32" },
    ]);

    // Hash without scalarsByNode parameter
    const hash1 = hashIRGraph(graph);
    // Hash with empty scalarsByNode map
    const hash2 = hashIRGraph(graph, new Map());
    // Should be identical
    expect(hash1).toBe(hash2);
  });

  it("scalarsByNode parameter overrides node.scalarValues", () => {
    const graph = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [2], dtype: "f32" },
      { id: 1, op: "mul", epoch: 1, kind: "lazy_op", inputs: [0], shape: [2], dtype: "f32", scalarValues: [2.0] },
    ]);

    // Hash using node.scalarValues (2.0)
    const hashFromNode = hashIRGraph(graph);

    // Hash using scalarsByNode override (3.0)
    const override = new Map<number, number[]>();
    override.set(1, [3.0]);
    const hashFromMap = hashIRGraph(graph, override);

    // Should differ because scalar value is different
    expect(hashFromNode).not.toBe(hashFromMap);
  });

  it("hash is deterministic across calls", () => {
    const graph = makeGraph([
      { id: 0, op: "input", epoch: 1, kind: "lazy_op", inputs: [], shape: [2, 3], dtype: "f32" },
      { id: 1, op: "mul", epoch: 1, kind: "lazy_op", inputs: [0], shape: [2, 3], dtype: "f32", scalarValues: [42.5] },
    ]);

    const hash1 = hashIRGraph(graph);
    const hash2 = hashIRGraph(graph);
    const hash3 = hashIRGraph(graph);

    expect(hash1).toBe(hash2);
    expect(hash2).toBe(hash3);
  });
});
