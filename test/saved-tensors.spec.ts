import { describe, expect, it } from "vitest";
import { Engine, SavedTensorModifiedError } from "../src";

describe("saved-for-backward guards", () => {
  it("throws if base is mutated after save", () => {
    const engine = new Engine();
    const saved = engine._debug_saveForBackward(4);

    engine._debug_baseCommit(4, 20);

    expect(() => engine._debug_useSavedTensor(saved)).toThrow(
      SavedTensorModifiedError,
    );
  });

  it("allows use when no mutation occurred", () => {
    const engine = new Engine();
    const saved = engine._debug_saveForBackward(2);

    expect(() => engine._debug_useSavedTensor(saved)).not.toThrow();
  });
});

describe("publish-save is token-linear", () => {
  it("advances tokGlobal and records a semantic event", () => {
    const engine = new Engine();
    const before = engine._debugSnapshot().tokGlobal.id;

    engine._debug_publishSave(1);

    const after = engine._debugSnapshot().tokGlobal.id;
    expect(after).not.toBe(before);

    const effects = engine.trace
      .snapshot()
      .filter((event) => event.type === "effect");
    expect(effects).toContainEqual({
      type: "effect",
      op: "publish_save",
      input: before,
      output: after,
    });
  });
});
