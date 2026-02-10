import { describe, expect, it } from "vitest";
import { Engine, EngineBusyError } from "../src";

describe("exec lock", () => {
  it("throws on overlapping entrypoints and releases the lock", () => {
    const engine = new Engine();

    expect(() =>
      engine._debug_runEntryPoint(() => {
        engine._debug_runEntryPoint(() => undefined);
      }),
    ).toThrow(EngineBusyError);

    expect(() => engine._debug_runEntryPoint(() => undefined)).not.toThrow();
  });

  it("drains finalizeQueue at entry and exit even when throwing", () => {
    const engine = new Engine();
    engine._debug_enqueueFinalize({ id: 1 });

    expect(() =>
      engine._debug_runEntryPoint(() => {
        throw new Error("boom");
      }),
    ).toThrow("boom");

    const drains = engine.trace
      .snapshot()
      .filter((event) => event.type === "finalize_drain");

    expect(drains).toEqual([
      { type: "finalize_drain", count: 1 },
      { type: "finalize_drain", count: 0 },
    ]);
  });

  it("allows cleanup-only drains while busy or poisoned", () => {
    const engine = new Engine();
    engine._debug_enqueueFinalize({ id: 2 });

    engine._debug_runEntryPoint(() => {
      expect(() => engine._debug_drainFinalizeQueueCleanupOnly()).not.toThrow();
    });

    engine._debug_enqueueFinalize({ id: 3 });
    engine._debug_poison();
    expect(() => engine._debug_drainFinalizeQueueCleanupOnly()).not.toThrow();
  });

  it("holds the lock across async entrypoints", async () => {
    const engine = new Engine();
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
