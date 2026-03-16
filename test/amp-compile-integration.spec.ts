import { describe, expect, it } from "vitest";
import { createAutocastContext, pushAutocast } from "../src/engine/amp";
import { RuntimeEngine } from "../src/runtime/engine";

describe("AMP Compile Integration (§12)", () => {
  it("engine has setAutocastContext method", () => {
    const engine = new RuntimeEngine();
    expect(typeof engine.setAutocastContext).toBe("function");
    expect(typeof engine.getAutocastContext).toBe("function");
  });

  it("autocast context is null by default", () => {
    const engine = new RuntimeEngine();
    expect(engine.getAutocastContext()).toBeNull();
  });

  it("can set and get autocast context", () => {
    const engine = new RuntimeEngine();
    const ctx = createAutocastContext();
    pushAutocast(ctx, { enabled: true });

    engine.setAutocastContext(ctx);
    expect(engine.getAutocastContext()).toBe(ctx);
    expect(engine.getAutocastContext()?.current.enabled).toBe(true);
  });

  it("can clear autocast context", () => {
    const engine = new RuntimeEngine();
    const ctx = createAutocastContext();
    pushAutocast(ctx, { enabled: true });

    engine.setAutocastContext(ctx);
    expect(engine.getAutocastContext()).not.toBeNull();

    engine.setAutocastContext(null);
    expect(engine.getAutocastContext()).toBeNull();
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
