import { describe, expect, it } from "vitest";
import { Engine, SavedTensorModifiedError } from "../src/engine/engine";

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
