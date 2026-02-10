import { describe, expect, it } from "vitest";
import {
  AsyncInCompileError,
  Engine,
  HostReadInCompileError,
  InvalidTraceTensorEscapeError,
  type TraceTensor,
} from "../src";

describe("compile staging restrictions", () => {
  it("throws on host reads during compile", () => {
    const engine = new Engine();
    const compiled = engine.compile(() => {
      engine._debug_hostRead();
      return undefined;
    });

    expect(() => compiled()).toThrow(HostReadInCompileError);
  });

  it("rejects async work during compile", () => {
    const engine = new Engine();
    const compiled = engine.compile(async () => 1);

    expect(() => compiled()).toThrow(AsyncInCompileError);
  });

  it("marks non-returned trace tensors as stale", () => {
    const engine = new Engine();
    let escaped: TraceTensor | undefined;

    const compiled = engine.compile(() => {
      escaped = engine._debug_emitLazyOp("hidden");
      return engine._debug_emitLazyOp("result");
    });

    const result = compiled();
    if (!escaped) {
      throw new Error("Expected a trace tensor to escape");
    }
    expect(() => engine._debug_useTraceTensor(result)).not.toThrow();
    expect(() => engine._debug_useTraceTensor(escaped)).toThrow(
      InvalidTraceTensorEscapeError,
    );
  });

  it("emits lazy nodes without executing effects", () => {
    const engine = new Engine();
    const before = engine._debugSnapshot().tokGlobal.id;

    const compiled = engine.compile(() => engine._debug_emitLazyOp("noop"));
    compiled();

    const after = engine._debugSnapshot().tokGlobal.id;
    expect(after).toBe(before);

    const effects = engine.trace
      .snapshot()
      .filter((event) => event.type === "effect");
    const lazyOps = engine.trace
      .snapshot()
      .filter((event) => event.type === "lazy_op");

    expect(effects).toEqual([]);
    expect(lazyOps).toHaveLength(1);
  });
});
