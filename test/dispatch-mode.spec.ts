import { describe, expect, it } from "vitest";
import {
  type DispatchMode,
  RuntimeEngine,
  TidyDispatchMode,
} from "../src/runtime/engine";
import type { Tensor } from "../src/runtime/tensor";
import { Torchlette } from "../src/frontend";

describe("dispatch mode infrastructure", () => {
  it("push/pop basic stack operation", () => {
    const engine = new RuntimeEngine();
    const mode: DispatchMode = { onTensorCreated: () => {} };
    engine.pushDispatchMode(mode);
    const popped = engine.popDispatchMode();
    expect(popped).toBe(mode);
  });

  it("pop on empty stack throws", () => {
    const engine = new RuntimeEngine();
    expect(() => engine.popDispatchMode()).toThrow("No dispatch mode to pop");
  });

  it("onTensorCreated called for every runtime op", () => {
    const engine = new RuntimeEngine();
    const created: Tensor[] = [];
    const mode: DispatchMode = {
      onTensorCreated: (t) => created.push(t),
    };
    engine.pushDispatchMode(mode);

    const a = engine.tensorFromArray([1, 2], [2]);
    const b = engine.tensorFromArray([3, 4], [2]);
    const c = engine.add(a, b);

    expect(created.length).toBe(3);
    expect(created[0]).toBe(a);
    expect(created[1]).toBe(b);
    expect(created[2]).toBe(c);

    engine.popDispatchMode();
  });

  it("nested modes — both receive notifications independently", () => {
    const engine = new RuntimeEngine();
    const outerCreated: Tensor[] = [];
    const innerCreated: Tensor[] = [];

    const outer: DispatchMode = {
      onTensorCreated: (t) => outerCreated.push(t),
    };
    const inner: DispatchMode = {
      onTensorCreated: (t) => innerCreated.push(t),
    };

    engine.pushDispatchMode(outer);
    engine.tensorFromArray([1], [1]); // seen by outer only

    engine.pushDispatchMode(inner);
    engine.tensorFromArray([2], [1]); // seen by both

    engine.popDispatchMode(); // pop inner
    engine.tensorFromArray([3], [1]); // seen by outer only
    engine.popDispatchMode(); // pop outer

    expect(outerCreated.length).toBe(3);
    expect(innerCreated.length).toBe(1);
  });

  it("onTensorEscaped called by markEscaped", () => {
    const engine = new RuntimeEngine();
    const escaped: Tensor[] = [];
    const mode: DispatchMode = {
      onTensorCreated: () => {},
      onTensorEscaped: (t) => escaped.push(t),
    };
    engine.pushDispatchMode(mode);

    const t = engine.tensorFromArray([1], [1]);
    expect(escaped.length).toBe(0);

    engine.markEscaped(t);
    expect(escaped.length).toBe(1);
    expect(escaped[0]).toBe(t);

    engine.popDispatchMode();
  });

  it("markEscaped notifies all active modes", () => {
    const engine = new RuntimeEngine();
    const escaped1: Tensor[] = [];
    const escaped2: Tensor[] = [];

    engine.pushDispatchMode({
      onTensorCreated: () => {},
      onTensorEscaped: (t) => escaped1.push(t),
    });
    engine.pushDispatchMode({
      onTensorCreated: () => {},
      onTensorEscaped: (t) => escaped2.push(t),
    });

    const t = engine.tensorFromArray([1], [1]);
    engine.markEscaped(t);

    expect(escaped1.length).toBe(1);
    expect(escaped2.length).toBe(1);

    engine.popDispatchMode();
    engine.popDispatchMode();
  });

  it("startIntermediateTracking/stopIntermediateTracking backward compat", () => {
    const engine = new RuntimeEngine();
    engine.startIntermediateTracking();

    const a = engine.tensorFromArray([1, 2], [2]);
    const b = engine.tensorFromArray([3, 4], [2]);

    const tracked = engine.stopIntermediateTracking();
    expect(tracked.size).toBe(2);
    expect(tracked.has(a)).toBe(true);
    expect(tracked.has(b)).toBe(true);
  });

  it("nested intermediate tracking works", () => {
    const engine = new RuntimeEngine();

    engine.startIntermediateTracking();
    const a = engine.tensorFromArray([1], [1]);

    engine.startIntermediateTracking();
    const b = engine.tensorFromArray([2], [1]);
    const innerTracked = engine.stopIntermediateTracking();

    const c = engine.tensorFromArray([3], [1]);
    const outerTracked = engine.stopIntermediateTracking();

    // Inner scope only sees tensor created during its lifetime
    expect(innerTracked.size).toBe(1);
    expect(innerTracked.has(b)).toBe(true);

    // Outer scope sees all three (it was active the whole time)
    expect(outerTracked.size).toBe(3);
    expect(outerTracked.has(a)).toBe(true);
    expect(outerTracked.has(b)).toBe(true);
    expect(outerTracked.has(c)).toBe(true);
  });
});

describe("TidyDispatchMode", () => {
  it("auto-disposes unwrapped intermediates within tidy", () => {
    const api = new Torchlette();

    // Track what the runtime creates
    const created: Tensor[] = [];
    const mode: DispatchMode = {
      onTensorCreated: (t) => created.push(t),
    };
    api.runtime.pushDispatchMode(mode);

    const result = api.tidy(() => {
      const a = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = api.tensorFromArray([5, 6, 7, 8], [2, 2]);
      const c = a.add(b);
      return c;
    });

    api.runtime.popDispatchMode();

    // The runtime tensors backing a and b should be disposed
    // (they were wrapped, but their engine tensors were disposed by engine tidy)
    // Result tensor should survive
    expect(result).toBeDefined();
    expect(result._unwrap().disposed).toBe(false);
  });

  it("disposeNonEscaped cleans up tracked non-escaped tensors", () => {
    const mode = new TidyDispatchMode();
    const engine = new RuntimeEngine();

    engine.pushDispatchMode(mode);
    const a = engine.tensorFromArray([1], [1]);
    const b = engine.tensorFromArray([2], [1]);

    // Mark 'a' as escaped (e.g. it was wrapped)
    engine.markEscaped(a);
    engine.popDispatchMode();

    mode.disposeNonEscaped();

    // 'b' should be disposed (not escaped), 'a' should survive
    expect(a.disposed).toBe(false);
    expect(b.disposed).toBe(true);
  });

  it("disposeNonEscaped skips already-disposed tensors", () => {
    const mode = new TidyDispatchMode();
    const engine = new RuntimeEngine();

    engine.pushDispatchMode(mode);
    const t = engine.tensorFromArray([1], [1]);
    engine.popDispatchMode();

    // Manually dispose first
    t.dispose();
    expect(t.disposed).toBe(true);

    // Should not throw
    mode.disposeNonEscaped();
  });

  it("nested tidy scopes work correctly", () => {
    const api = new Torchlette();

    const outer = api.tidy(() => {
      const a = api.tensorFromArray([1, 2], [2]);
      const inner = api.tidy(() => {
        const b = api.tensorFromArray([3, 4], [2]);
        return a.add(b);
      });
      // b's runtime tensor should be cleaned up by inner tidy
      return inner;
    });

    // Result should survive both tidy scopes
    expect(outer).toBeDefined();
    expect(outer._unwrap().disposed).toBe(false);
  });
});
