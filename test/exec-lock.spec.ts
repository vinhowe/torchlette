import { describe, expect, it } from "vitest";
import { EngineBusyError, RuntimeEngine } from "../src/runtime/engine";

describe("exec lock", () => {
  it("throws on overlapping entrypoints and releases the lock", () => {
    const engine = new RuntimeEngine();

    expect(() =>
      engine._debug_runEntryPoint(() => {
        engine._debug_runEntryPoint(() => undefined);
      }),
    ).toThrow(EngineBusyError);

    expect(() => engine._debug_runEntryPoint(() => undefined)).not.toThrow();
  });

  it("holds the lock across async entrypoints", async () => {
    const engine = new RuntimeEngine();
    let resolve: (() => void) | undefined;

    const pending = engine.runEntryPoint(
      async () =>
        await new Promise<void>((next) => {
          resolve = next;
        }),
    );

    await expect(engine.runEntryPoint(async () => undefined)).rejects.toThrow(
      EngineBusyError,
    );

    resolve?.();
    await pending;

    await expect(engine.runEntryPoint(async () => undefined)).resolves.toBe(
      undefined,
    );
  });
});
