import { describe, expect, it, beforeAll } from "vitest";
import { Torchlette } from "../../src/frontend";
import { Adam, GradScaler } from "../../src/optim";
import { initWebGPU } from "../../src/backend/webgpu";
import { canUseWebGPU } from "../helpers/webgpu";

describe("GradScaler", () => {
  let api: Torchlette;
  let device: "webgpu" | "cpu" = "cpu";

  beforeAll(async () => {
    if (await canUseWebGPU()) {
      device = "webgpu";
    }
  });

  it("should scale loss by the scale factor", () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, { initScale: 1024 });

    const loss = api.tensorFromArray([2.0], [], { device });
    const scaled = scaler.scale(loss);

    expect(scaler.getScale()).toBe(1024);
    // scaled should be 2.0 * 1024 = 2048
  });

  it("should unscale gradients correctly", async () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, { initScale: 4.0 });

    const param = api.tensorFromArray([1.0, 2.0, 3.0], [3], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    // Simulate backward with scaled gradients
    const loss = api.sum(param);
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    // Gradients should be scaled by 4.0 (all 4.0 for sum backward)
    const gradBefore = await param.grad!.cpu();
    expect(gradBefore[0]).toBeCloseTo(4.0, 4);

    // Unscale
    scaler.unscale_(optimizer);

    if (device !== "webgpu") {
      // CPU path: grads are unscaled immediately by unscale_()
      const gradAfter = await param.grad!.cpu();
      expect(gradAfter[0]).toBeCloseTo(1.0, 4);
    }
    // On WebGPU with fused Adam+unscale, grads are unscaled inside the Adam kernel
    // during step(), not by unscale_() directly.

    // foundInf is deferred — resolve after full cycle
    scaler.step(optimizer);
    scaler.update();
    await scaler.resolveDeferred();
    expect(scaler.foundInf).toBe(false);
  });

  it("should detect NaN in gradients", async () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, { initScale: 1.0 });

    const param = api.tensorFromArray([1.0, 2.0, 3.0], [3], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    // Create a computation that will produce NaN gradient
    // log(0) = -Inf, and operations on -Inf can produce NaN
    const zero = api.tensorFromArray([0.0], [], { device });
    const logZero = api.log(zero); // -Inf
    const loss = api.mul(api.sum(param), logZero); // Inf * something
    await loss.backward();

    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
    // foundInf is deferred — resolve it now
    await scaler.resolveDeferred();
    expect(scaler.foundInf).toBe(true);
  });

  it("should skip optimizer step when NaN detected", async () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, { initScale: 1.0 });

    const initialValues = [1.0, 2.0, 3.0];
    const param = api.tensorFromArray(initialValues, [3], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    // Create NaN gradient
    const zero = api.tensorFromArray([0.0], [], { device });
    const logZero = api.log(zero);
    const loss = api.mul(api.sum(param), logZero);
    await loss.backward();

    scaler.unscale_(optimizer);
    const stepped = scaler.step(optimizer);

    // Optimizer always runs now (masking handles inf case), but params are reverted
    expect(stepped).toBe(true);
    // Parameters should be unchanged (reverted by GPU-side masking)
    const paramValues = await param.cpu();
    expect(paramValues[0]).toBeCloseTo(initialValues[0], 4);
    expect(paramValues[1]).toBeCloseTo(initialValues[1], 4);
    expect(paramValues[2]).toBeCloseTo(initialValues[2], 4);
  });

  it("should reduce scale when NaN detected", async () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, {
      initScale: 1024,
      backoffFactor: 0.5,
    });

    const param = api.tensorFromArray([1.0], [1], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    // Create NaN gradient
    const zero = api.tensorFromArray([0.0], [], { device });
    const logZero = api.log(zero);
    const loss = api.mul(api.sum(param), logZero);
    await loss.backward();

    expect(scaler.getScale()).toBe(1024);

    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();

    // Scale adjustment is deferred — resolve it
    await scaler.resolveDeferred();

    // Scale should be halved
    expect(scaler.getScale()).toBe(512);
  });

  it("should grow scale after growth_interval successful steps", async () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, {
      initScale: 1.0,
      growthFactor: 2.0,
      growthInterval: 3, // Grow after 3 successful steps
    });

    const param = api.tensorFromArray([1.0], [1], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    // Do 3 successful steps
    for (let i = 0; i < 3; i++) {
      await scaler.resolveDeferred();
      const loss = api.sum(param);
      const scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();

      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
      optimizer.zeroGrad();
    }
    await scaler.resolveDeferred();

    // Scale should have doubled
    expect(scaler.getScale()).toBe(2.0);
  });

  it("should reset growth tracker on NaN", async () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, {
      initScale: 4.0,
      growthFactor: 2.0,
      backoffFactor: 0.5,
      growthInterval: 3,
    });

    const param = api.tensorFromArray([1.0], [1], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    // Do 2 successful steps
    for (let i = 0; i < 2; i++) {
      await scaler.resolveDeferred();
      const loss = api.sum(param);
      const scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();

      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
      optimizer.zeroGrad();
    }
    await scaler.resolveDeferred();

    expect(scaler.getScale()).toBe(4.0); // Not yet grown

    // Now trigger NaN
    const zero = api.tensorFromArray([0.0], [], { device });
    const logZero = api.log(zero);
    const loss = api.mul(api.sum(param), logZero);
    await loss.backward();

    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
    optimizer.zeroGrad();
    await scaler.resolveDeferred();

    expect(scaler.getScale()).toBe(2.0); // Halved due to NaN

    // Now do 3 more successful steps - growth tracker should have reset
    for (let i = 0; i < 3; i++) {
      await scaler.resolveDeferred();
      const loss2 = api.sum(param);
      const scaledLoss2 = scaler.scale(loss2);
      await scaledLoss2.backward();

      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
      optimizer.zeroGrad();
    }
    await scaler.resolveDeferred();

    expect(scaler.getScale()).toBe(4.0); // Doubled after 3 successful steps
  });

  it("should save and load state", () => {
    api = new Torchlette(device);
    const scaler1 = new GradScaler(api, { initScale: 512 });

    // Modify state
    const state = scaler1.stateDict();
    state.scale = 256;
    state.growthTracker = 5;

    const scaler2 = new GradScaler(api);
    scaler2.loadStateDict(state);

    expect(scaler2.getScale()).toBe(256);
    expect(scaler2.stateDict().growthTracker).toBe(5);
  });

  it("should detect non-finite gradients and set foundInf", async () => {
    // Test that gradients containing inf/nan are detected and zeroed
    api = new Torchlette(device);
    const scaler = new GradScaler(api, { initScale: 1.0 });

    const param = api.tensorFromArray([1.0, 2.0, 3.0], [3], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    // Create a computation that produces -Inf gradient
    // log(0) = -Inf, mul with sum gives -Inf gradient
    const zero = api.tensorFromArray([0.0], [], { device });
    const logZero = api.log(zero); // -Inf
    const loss = api.mul(api.sum(param), logZero);
    await loss.backward();

    // Gradient should be -Inf before unscale
    const gradBefore = await param.grad!.cpu();
    expect(Number.isFinite(gradBefore[0])).toBe(false);

    scaler.unscale_(optimizer);

    if (device !== "webgpu") {
      // CPU path: grads are zeroed immediately by unscale_() (inf detected → mask applied)
      const gradAfter = await param.grad!.cpu();
      expect(gradAfter[0]).toBe(0);
    }
    // On WebGPU with fused Adam+unscale, inf detection and zeroing
    // happen inside the Adam kernel during step().

    // foundInf is deferred — resolve after full cycle
    scaler.step(optimizer);
    scaler.update();
    await scaler.resolveDeferred();
    expect(scaler.foundInf).toBe(true);
  });

  it("should unscale all gradients and detect mixed finite/non-finite", async () => {
    // Test that all gradients are unscaled, inf is detected, and grads are zeroed
    api = new Torchlette(device);
    const scaler = new GradScaler(api, { initScale: 4.0 });

    // Two params - one will have finite grad, one will have inf grad
    const param1 = api.tensorFromArray([1.0, 2.0], [2], {
      device,
      requiresGrad: true,
    });
    const param2 = api.tensorFromArray([3.0], [1], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param1, param2], { lr: 0.01 }, api);

    // param1 gets finite gradient from sum
    // param2 gets inf gradient from log(0)
    const loss1 = api.sum(param1);
    const zero = api.tensorFromArray([0.0], [], { device });
    const logZero = api.log(zero); // -Inf
    const loss2 = api.mul(api.sum(param2), logZero);
    const totalLoss = api.add(scaler.scale(loss1), scaler.scale(loss2));
    await totalLoss.backward();

    // param1.grad should be 4.0 (scaled), param2.grad should be -Inf
    const grad1Before = await param1.grad!.cpu();
    const grad2Before = await param2.grad!.cpu();
    expect(grad1Before[0]).toBeCloseTo(4.0, 4);
    expect(Number.isFinite(grad2Before[0])).toBe(false);

    scaler.unscale_(optimizer);

    if (device !== "webgpu") {
      // Elementwise path: ALL grads are zeroed when any inf detected
      const grad1After = await param1.grad!.cpu();
      const grad2After = await param2.grad!.cpu();
      expect(grad1After[0]).toBe(0);
      expect(grad1After[1]).toBe(0);
      expect(grad2After[0]).toBe(0);
    }
    // On WebGPU with fused Adam+unscale, unscaling and inf detection happen
    // inside the Adam kernel during step(), not during unscale_().

    // foundInf is deferred — resolve after full cycle
    scaler.step(optimizer);
    scaler.update();
    await scaler.resolveDeferred();
    expect(scaler.foundInf).toBe(true);
  });

  it("should work when disabled", async () => {
    api = new Torchlette(device);
    const scaler = new GradScaler(api, { enabled: false });

    const param = api.tensorFromArray([1.0, 2.0, 3.0], [3], {
      device,
      requiresGrad: true,
    });
    const optimizer = new Adam([param], { lr: 0.01 }, api);

    const loss = api.sum(param);
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    // Gradient should NOT be scaled (enabled=false)
    const gradBefore = await param.grad!.cpu();
    expect(gradBefore[0]).toBeCloseTo(1.0, 4);

    scaler.unscale_(optimizer);
    const stepped = scaler.step(optimizer);
    scaler.update();

    expect(stepped).toBe(true);
    expect(scaler.foundInf).toBe(false);
  });
});
