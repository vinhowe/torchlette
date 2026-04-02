import { describe, expect, it } from "vitest";

import {
  type EngineTensor,
  PoisonedEngineError,
  RuntimeEngine,
} from "../src/runtime/engine";

describe("lifecycle: tidy/keep/dispose", () => {
  it("disposes tensors created in tidy unless kept or returned", () => {
    const engine = new RuntimeEngine();
    let kept: EngineTensor | null = null;
    let dropped: EngineTensor | null = null;

    const returned = engine.tidy(() => {
      const a = engine.createTensor();
      const b = engine.createTensor();
      const c = engine.createTensor();

      engine.keep(b);
      kept = b;
      dropped = c;

      return a;
    });

    expect(returned.disposed).toBe(false);
    expect(returned.escapes).toBe(true);
    expect(kept?.disposed).toBe(false);
    expect(kept?.escapes).toBe(true);
    expect(dropped?.disposed).toBe(true);

    expect(engine._debug_getBasePinCount(returned.baseId)).toBe(1);
    expect(engine._debug_getBasePinCount(kept?.baseId ?? 0)).toBe(1);
    expect(engine._debug_getBasePinCount(dropped?.baseId ?? 0)).toBe(0);
  });

  it("allows cleanup-only disposal while busy or poisoned", () => {
    const engine = new RuntimeEngine();
    const tensor = engine.createTensor();

    engine._debug_runEntryPoint(() => {
      expect(() => engine.dispose(tensor)).not.toThrow();
    });

    const tensor2 = engine.createTensor();
    engine._debug_poison();
    expect(() => engine.dispose(tensor2)).not.toThrow();
  });
});

describe("lifecycle: markStep", () => {
  it("does not throw when poisoned (markStep acquires lock)", async () => {
    const engine = new RuntimeEngine();
    engine._debug_poison();

    await expect(engine.markStep()).rejects.toThrow(PoisonedEngineError);
  });
});
