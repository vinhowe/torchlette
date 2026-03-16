import { describe, expect, it } from "vitest";
import { Engine, PoisonedEngineError } from "../src/engine/engine";

describe("poisoning", () => {
  it("throws on execution-affecting ops after poison", () => {
    const engine = new Engine();
    engine._debug_poison();

    expect(() => engine._debug_runEntryPoint(() => undefined)).toThrow(
      PoisonedEngineError,
    );
  });

  it("allows disposal when poisoned", () => {
    const engine = new Engine();
    const tensor = engine.createTensor();
    engine._debug_poison();
    expect(() => engine.dispose(tensor)).not.toThrow();
  });
});
