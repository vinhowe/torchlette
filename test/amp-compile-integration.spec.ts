import { describe, expect, it } from "vitest";

import { Engine } from "../src/engine/engine";
import { createAutocastContext, pushAutocast } from "../src/engine/amp";

describe("AMP Compile Integration (§12)", () => {
  it("engine has setAutocastContext method", () => {
    const engine = new Engine();
    expect(typeof engine.setAutocastContext).toBe("function");
    expect(typeof engine.getAutocastContext).toBe("function");
  });

  it("autocast context is null by default", () => {
    const engine = new Engine();
    expect(engine.getAutocastContext()).toBeNull();
  });

  it("can set and get autocast context", () => {
    const engine = new Engine();
    const ctx = createAutocastContext();
    pushAutocast(ctx, { enabled: true });

    engine.setAutocastContext(ctx);
    expect(engine.getAutocastContext()).toBe(ctx);
    expect(engine.getAutocastContext()?.current.enabled).toBe(true);
  });

  it("can clear autocast context", () => {
    const engine = new Engine();
    const ctx = createAutocastContext();
    pushAutocast(ctx, { enabled: true });

    engine.setAutocastContext(ctx);
    expect(engine.getAutocastContext()).not.toBeNull();

    engine.setAutocastContext(null);
    expect(engine.getAutocastContext()).toBeNull();
  });

  describe("compile with AMP", () => {
    it("compiled graph includes AMP policy hash in cache key", () => {
      const engine = new Engine();
      const ctx = createAutocastContext();
      pushAutocast(ctx, { enabled: true });
      engine.setAutocastContext(ctx);

      // Create a simple compiled function that does a matmul
      const compiled = engine.compile(() => {
        const a = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        const b = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        return engine._debug_emitLazyOp("matmul", {
          inputs: [a, b],
          shape: [2, 2],
          dtype: "f32",
        });
      });

      compiled();

      const cacheKey = engine._debug_getLastCacheKey();
      expect(cacheKey).not.toBeNull();
      // Cache key should include AMP policy info
      expect(cacheKey?.irHash).toContain("amp=");
    });

    it("different AMP settings produce different cache keys", () => {
      const engine = new Engine();

      // First compile without AMP
      const compiled1 = engine.compile(() => {
        const a = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        const b = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        return engine._debug_emitLazyOp("matmul", {
          inputs: [a, b],
          shape: [2, 2],
          dtype: "f32",
        });
      });
      compiled1();
      const key1 = engine._debug_getLastCacheKey();

      // Now compile with AMP enabled
      const ctx = createAutocastContext();
      pushAutocast(ctx, { enabled: true });
      engine.setAutocastContext(ctx);

      const compiled2 = engine.compile(() => {
        const a = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        const b = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        return engine._debug_emitLazyOp("matmul", {
          inputs: [a, b],
          shape: [2, 2],
          dtype: "f32",
        });
      });
      compiled2();
      const key2 = engine._debug_getLastCacheKey();

      expect(key1?.irHash).not.toBe(key2?.irHash);
      expect(key1?.irHash).toContain("amp=disabled");
      expect(key2?.irHash).toContain("amp=f16");
    });

    it("AMP transform modifies graph for matmul", () => {
      const engine = new Engine();
      const ctx = createAutocastContext();
      pushAutocast(ctx, { enabled: true });
      engine.setAutocastContext(ctx);

      const compiled = engine.compile(() => {
        const a = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        const b = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        return engine._debug_emitLazyOp("matmul", {
          inputs: [a, b],
          shape: [2, 2],
          dtype: "f32",
        });
      });

      compiled();

      const graph = engine._debug_getLastCompiledGraph();
      expect(graph).not.toBeNull();

      // Should have cast nodes inserted
      const castNodes = graph!.nodes.filter((n) => n.op === "cast");
      expect(castNodes.length).toBeGreaterThan(0);

      // Matmul should output f16 (per AMP policy)
      const matmulNode = graph!.nodes.find((n) => n.op === "matmul");
      expect(matmulNode?.dtype).toBe("f16");
    });

    it("AMP transform does not modify non-eligible ops", () => {
      const engine = new Engine();
      const ctx = createAutocastContext();
      pushAutocast(ctx, { enabled: true });
      engine.setAutocastContext(ctx);

      const compiled = engine.compile(() => {
        const a = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        const b = engine._debug_emitLazyOp("input", {
          shape: [2, 2],
          dtype: "f32",
        });
        return engine._debug_emitLazyOp("add", {
          inputs: [a, b],
          shape: [2, 2],
          dtype: "f32",
        });
      });

      compiled();

      const graph = engine._debug_getLastCompiledGraph();
      expect(graph).not.toBeNull();

      // No cast nodes needed for add (not f16-eligible)
      const castNodes = graph!.nodes.filter((n) => n.op === "cast");
      expect(castNodes.length).toBe(0);
    });
  });
});

describe("Frontend AMP-Engine Integration", () => {
  it("autocast sets engine context", async () => {
    const { Torchlette } = await import("../src/frontend");
    const torch = new Torchlette("cpu");

    // Engine context should be null before autocast
    expect(torch._getAutocastContext().current.enabled).toBe(false);

    let wasEnabled = false;
    torch.autocast(() => {
      wasEnabled = torch.isAutocastEnabled;
    });

    expect(wasEnabled).toBe(true);
    // After autocast, should be disabled again
    expect(torch.isAutocastEnabled).toBe(false);
  });

  it("nested autocast updates engine context", async () => {
    const { Torchlette } = await import("../src/frontend");
    const torch = new Torchlette("cpu");

    const states: boolean[] = [];

    torch.autocast(() => {
      states.push(torch.isAutocastEnabled); // true
      torch.autocast(
        () => {
          states.push(torch.isAutocastEnabled); // false (disabled)
        },
        { enabled: false },
      );
      states.push(torch.isAutocastEnabled); // true again
    });

    expect(states).toEqual([true, false, true]);
  });
});
