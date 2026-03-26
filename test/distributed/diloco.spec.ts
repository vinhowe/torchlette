import { describe, expect, it } from "vitest";
import { DiLoCoTrainer } from "../../src/distributed/diloco";
import { NesterovOuterOptimizer } from "../../src/distributed/outer-optimizer";
import { Torchlette } from "../../src/frontend/torchlette";

describe("NesterovOuterOptimizer", () => {
  it("applies pseudo-gradient to parameters", async () => {
    const api = new Torchlette("cpu");
    const param = api.tensorFromArray([1, 2, 3], [3]);
    param.requires_grad_(true);

    const optimizer = new NesterovOuterOptimizer(api, { lr: 1.0, momentum: 0 });

    // Pseudo-gradient: [0.1, 0.2, 0.3]
    const delta = api.tensorFromArray([0.1, 0.2, 0.3], [3]);

    optimizer.step([param], [delta]);

    // With lr=1, mu=0: theta = theta + lr * delta = [1.1, 2.2, 3.3]
    const result = await param.cpu();
    expect(result[0]).toBeCloseTo(1.1, 5);
    expect(result[1]).toBeCloseTo(2.2, 5);
    expect(result[2]).toBeCloseTo(3.3, 5);

    optimizer.dispose();
  });

  it("accumulates momentum across steps", async () => {
    const api = new Torchlette("cpu");
    const param = api.tensorFromArray([10], [1]);
    param.requires_grad_(true);

    const optimizer = new NesterovOuterOptimizer(api, {
      lr: 1.0,
      momentum: 0.9,
    });

    // Step 1: delta = 1.0
    // v = 0.9*0 + 1.0 = 1.0
    // theta = 10 + 1.0*1.0 = 11.0
    optimizer.step([param], [api.tensorFromArray([1.0], [1])]);
    expect((await param.cpu())[0]).toBeCloseTo(11.0, 5);

    // Step 2: delta = 1.0
    // v = 0.9*1.0 + 1.0 = 1.9
    // theta = 11.0 + 1.0*1.9 = 12.9
    optimizer.step([param], [api.tensorFromArray([1.0], [1])]);
    expect((await param.cpu())[0]).toBeCloseTo(12.9, 5);

    optimizer.dispose();
  });

  it("uses default DiLoCo hyperparameters", async () => {
    const api = new Torchlette("cpu");
    const optimizer = new NesterovOuterOptimizer(api);
    // Should not throw — defaults are lr=0.7, mu=0.9
    const param = api.tensorFromArray([1], [1]);
    param.requires_grad_(true);
    optimizer.step([param], [api.tensorFromArray([0.1], [1])]);
    // theta = 1 + 0.7 * (0.9*0 + 0.1) = 1 + 0.07 = 1.07
    expect((await param.cpu())[0]).toBeCloseTo(1.07, 5);
    optimizer.dispose();
  });
});

describe("DiLoCoTrainer", () => {
  it("computes pseudo-gradients after inner steps", async () => {
    const api = new Torchlette("cpu");
    const param = api.tensorFromArray([1, 2, 3], [3]);
    param.requires_grad_(true);

    const trainer = new DiLoCoTrainer(api, [param], { innerSteps: 10 });

    // Snapshot global params
    await trainer.snapshotGlobalParams();

    // Simulate inner training by modifying params directly
    api.copy_(param, api.tensorFromArray([1.1, 2.2, 3.3], [3]));

    // Pseudo-gradients should be local - global = [0.1, 0.2, 0.3]
    const pseudoGrads = await trainer.computePseudoGrads();
    expect(pseudoGrads.length).toBe(1);
    expect(pseudoGrads[0][0]).toBeCloseTo(0.1, 5);
    expect(pseudoGrads[0][1]).toBeCloseTo(0.2, 5);
    expect(pseudoGrads[0][2]).toBeCloseTo(0.3, 5);

    trainer.dispose();
  });

  it("outer step resets params to snapshot + outer update", async () => {
    const api = new Torchlette("cpu");
    const param = api.tensorFromArray([10, 20], [2]);
    param.requires_grad_(true);

    const trainer = new DiLoCoTrainer(api, [param], {
      innerSteps: 10,
      outerLR: 1.0,
      outerMomentum: 0,
    });

    // Snapshot: [10, 20]
    await trainer.snapshotGlobalParams();

    // Simulate inner training modifying params to [11, 22]
    api.copy_(param, api.tensorFromArray([11, 22], [2]));

    // Averaged pseudo-grad from "all workers": [1, 2]
    const avgDelta = api.tensorFromArray([1, 2], [2]);

    // Outer step: reset to snapshot [10, 20], then apply outer update
    // theta = [10, 20] + 1.0 * [1, 2] = [11, 22]
    trainer.outerStep([avgDelta]);

    const result = await param.cpu();
    expect(result[0]).toBeCloseTo(11, 5);
    expect(result[1]).toBeCloseTo(22, 5);

    trainer.dispose();
  });

  it("throws if pseudo-grads computed without snapshot", async () => {
    const api = new Torchlette("cpu");
    const param = api.tensorFromArray([1], [1]);
    param.requires_grad_(true);
    const trainer = new DiLoCoTrainer(api, [param]);

    await expect(trainer.computePseudoGrads()).rejects.toThrow(
      "snapshotGlobalParams",
    );
    trainer.dispose();
  });
});
