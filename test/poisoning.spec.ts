import { describe, expect, it } from "vitest";
import { Engine, PoisonedEngineError } from "../src";

describe("poisoning", () => {
  it("throws on execution-affecting ops after poison", () => {
    const engine = new Engine();
    engine._debug_poison();

    expect(() => engine.emitEffect("noop")).toThrow(PoisonedEngineError);
    expect(() => engine.orderedAccess(1, "load")).toThrow(PoisonedEngineError);
    expect(() => engine._debug_runEntryPoint(() => undefined)).toThrow(
      PoisonedEngineError,
    );
  });

  it("allows cleanup-only operations when poisoned", () => {
    const engine = new Engine();
    engine._debug_enqueueFinalize({ id: 1 });
    engine._debug_poison();

    expect(() => engine._debug_drainFinalizeQueueCleanupOnly()).not.toThrow();
  });
});
