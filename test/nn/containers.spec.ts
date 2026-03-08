/**
 * Tests for nn container modules: ModuleList, Sequential
 */
import { beforeEach, describe, expect, it } from "vitest";
import { type Tensor, Torchlette } from "../../src/frontend";
import { Linear } from "../../src/nn/linear";
import { Module } from "../../src/nn/module";
import { ModuleList } from "../../src/nn/modulelist";
import { Sequential } from "../../src/nn/sequential";

// Simple test module with one parameter and passthrough forward
class ScaleModule extends Module {
  declare readonly scale: Tensor;

  constructor(api: Torchlette, size: number) {
    super(api);
    this.registerParameter("scale", api.ones([size], { requiresGrad: true }));
  }

  forward(input: Tensor): Tensor {
    return this.api.mul(input, this.scale);
  }
}

// Module that adds a bias
class BiasModule extends Module {
  declare readonly bias: Tensor;

  constructor(api: Torchlette, size: number) {
    super(api);
    this.registerParameter("bias", api.ones([size], { requiresGrad: true }));
  }

  forward(input: Tensor): Tensor {
    return this.api.add(input, this.bias);
  }
}

// Parent module that holds containers as properties (for Proxy auto-registration tests)
class ParentModel extends Module {
  layers!: ModuleList;
  head!: Linear;

  constructor(api: Torchlette) {
    super(api);
    this.layers = new ModuleList(api, [
      new ScaleModule(api, 4),
      new BiasModule(api, 4),
    ]);
    this.head = new Linear(api, 4, 2);
  }

  forward(input: Tensor): Tensor {
    let x = input;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    return this.head.forward(x);
  }
}

class SeqParentModel extends Module {
  body!: Sequential;

  constructor(api: Torchlette) {
    super(api);
    this.body = new Sequential(
      api,
      new ScaleModule(api, 4),
      new BiasModule(api, 4),
    );
  }

  forward(input: Tensor): Tensor {
    return this.body.forward(input);
  }
}

describe("nn.ModuleList", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette("cpu");
  });

  it("starts empty when no modules provided", () => {
    const list = new ModuleList(api);
    expect(list.length).toBe(0);
  });

  it("initializes with provided modules", () => {
    const m1 = new ScaleModule(api, 4);
    const m2 = new BiasModule(api, 4);
    const list = new ModuleList(api, [m1, m2]);

    expect(list.length).toBe(2);
    expect(list.get(0)).toBe(m1);
    expect(list.get(1)).toBe(m2);
  });

  it("append() adds modules and returns this", () => {
    const list = new ModuleList(api);
    const m1 = new ScaleModule(api, 4);
    const m2 = new BiasModule(api, 4);

    const ret = list.append(m1);
    expect(ret).toBe(list);
    expect(list.length).toBe(1);
    expect(list.get(0)).toBe(m1);

    list.append(m2);
    expect(list.length).toBe(2);
    expect(list.get(1)).toBe(m2);
  });

  it("get(i) returns the correct module", () => {
    const modules = [
      new ScaleModule(api, 4),
      new BiasModule(api, 4),
      new ScaleModule(api, 8),
    ];
    const list = new ModuleList(api, modules);

    for (let i = 0; i < modules.length; i++) {
      expect(list.get(i)).toBe(modules[i]);
    }
  });

  it("supports iteration via for..of", () => {
    const m1 = new ScaleModule(api, 4);
    const m2 = new BiasModule(api, 4);
    const list = new ModuleList(api, [m1, m2]);

    const collected: Module[] = [];
    for (const mod of list) {
      collected.push(mod);
    }

    expect(collected).toHaveLength(2);
    expect(collected[0]).toBe(m1);
    expect(collected[1]).toBe(m2);
  });

  it("supports spread via Symbol.iterator", () => {
    const m1 = new ScaleModule(api, 4);
    const m2 = new BiasModule(api, 4);
    const list = new ModuleList(api, [m1, m2]);

    const collected = [...list];

    expect(collected).toHaveLength(2);
    expect(collected[0]).toBe(m1);
    expect(collected[1]).toBe(m2);
  });

  it("parameters() collects params from all children", () => {
    const m1 = new ScaleModule(api, 4); // 1 param: scale
    const m2 = new BiasModule(api, 4); // 1 param: bias
    const list = new ModuleList(api, [m1, m2]);

    const params = list.parameters();

    expect(params).toHaveLength(2);
    expect(params[0]).toBe(m1.scale);
    expect(params[1]).toBe(m2.bias);
  });

  it("parameters() collects params from nested Linear modules", () => {
    const l1 = new Linear(api, 4, 8); // weight + bias = 2 params
    const l2 = new Linear(api, 8, 2); // weight + bias = 2 params
    const list = new ModuleList(api, [l1, l2]);

    const params = list.parameters();

    expect(params).toHaveLength(4);
    expect(params[0]).toBe(l1.weight);
    expect(params[1]).toBe(l1.bias);
    expect(params[2]).toBe(l2.weight);
    expect(params[3]).toBe(l2.bias);
  });

  it("namedParameters() produces correct dotted paths", () => {
    const m1 = new ScaleModule(api, 4);
    const m2 = new BiasModule(api, 4);
    const list = new ModuleList(api, [m1, m2]);

    const named = list.namedParameters();

    expect(named).toHaveLength(2);
    expect(named[0][0]).toBe("0.scale");
    expect(named[0][1]).toBe(m1.scale);
    expect(named[1][0]).toBe("1.bias");
    expect(named[1][1]).toBe(m2.bias);
  });

  it("namedParameters() with prefix produces correct paths", () => {
    const m1 = new ScaleModule(api, 4);
    const list = new ModuleList(api, [m1]);

    const named = list.namedParameters("list");

    expect(named).toHaveLength(1);
    expect(named[0][0]).toBe("list.0.scale");
  });

  it("train() propagates to children", () => {
    const m1 = new ScaleModule(api, 4);
    const m2 = new BiasModule(api, 4);
    const list = new ModuleList(api, [m1, m2]);

    expect(m1.training).toBe(true);
    expect(m2.training).toBe(true);

    list.eval();
    expect(list.training).toBe(false);
    expect(m1.training).toBe(false);
    expect(m2.training).toBe(false);

    list.train();
    expect(list.training).toBe(true);
    expect(m1.training).toBe(true);
    expect(m2.training).toBe(true);
  });

  it("train(false) is equivalent to eval()", () => {
    const m1 = new ScaleModule(api, 4);
    const list = new ModuleList(api, [m1]);

    list.train(false);
    expect(list.training).toBe(false);
    expect(m1.training).toBe(false);
  });

  it("forward() throws", () => {
    const list = new ModuleList(api);
    const input = api.randn([2, 4]);

    expect(() => list.forward(input)).toThrow(
      "ModuleList does not implement forward()",
    );
  });
});

describe("nn.Sequential", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette("cpu");
  });

  it("forward() chains modules in order", async () => {
    // ScaleModule multiplies by 1 (ones), BiasModule adds 1 (ones)
    // Input [2,4] of twos -> scale by 1 -> [2,4] of twos -> add 1 -> [2,4] of threes
    const seq = new Sequential(
      api,
      new ScaleModule(api, 4),
      new BiasModule(api, 4),
    );

    const input = api.tensorFromArray([2, 2, 2, 2, 2, 2, 2, 2], [2, 4]);
    const output = seq.forward(input);

    expect(output.shape).toEqual([2, 4]);
    const data = await output.cpu();
    for (const v of data) {
      expect(v).toBeCloseTo(3.0); // 2 * 1 + 1 = 3
    }
  });

  it("forward() with single module", async () => {
    const seq = new Sequential(api, new BiasModule(api, 3));
    const input = api.zeros([1, 3]);
    const output = seq.forward(input);

    expect(output.shape).toEqual([1, 3]);
    const data = await output.cpu();
    for (const v of data) {
      expect(v).toBeCloseTo(1.0); // 0 + 1 = 1
    }
  });

  it("forward() with empty Sequential returns input unchanged", async () => {
    const seq = new Sequential(api);
    const input = api.tensorFromArray([1, 2, 3], [3]);
    const output = seq.forward(input);

    // With no children, the for..of loop body never executes; x stays as input
    expect(output).toBe(input);
  });

  it("parameters() collects params from all children", () => {
    const s = new ScaleModule(api, 4);
    const b = new BiasModule(api, 4);
    const seq = new Sequential(api, s, b);

    const params = seq.parameters();

    expect(params).toHaveLength(2);
    expect(params[0]).toBe(s.scale);
    expect(params[1]).toBe(b.bias);
  });

  it("parameters() with Linear children", () => {
    const l1 = new Linear(api, 4, 8);
    const l2 = new Linear(api, 8, 2);
    const seq = new Sequential(api, l1, l2);

    const params = seq.parameters();

    expect(params).toHaveLength(4); // 2 params per Linear
  });

  it("namedParameters() produces correct paths", () => {
    const s = new ScaleModule(api, 4);
    const b = new BiasModule(api, 4);
    const seq = new Sequential(api, s, b);

    const named = seq.namedParameters();

    expect(named).toHaveLength(2);
    expect(named[0][0]).toBe("0.scale");
    expect(named[0][1]).toBe(s.scale);
    expect(named[1][0]).toBe("1.bias");
    expect(named[1][1]).toBe(b.bias);
  });

  it("namedParameters() with prefix", () => {
    const s = new ScaleModule(api, 4);
    const seq = new Sequential(api, s);

    const named = seq.namedParameters("body");

    expect(named).toHaveLength(1);
    expect(named[0][0]).toBe("body.0.scale");
  });

  it("train()/eval() propagates to children", () => {
    const s = new ScaleModule(api, 4);
    const b = new BiasModule(api, 4);
    const seq = new Sequential(api, s, b);

    seq.eval();
    expect(seq.training).toBe(false);
    expect(s.training).toBe(false);
    expect(b.training).toBe(false);

    seq.train();
    expect(seq.training).toBe(true);
    expect(s.training).toBe(true);
    expect(b.training).toBe(true);
  });
});

describe("Proxy auto-registration", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette("cpu");
  });

  it("assigning ModuleList property auto-registers it", () => {
    const model = new ParentModel(api);

    // The ModuleList should be auto-registered via Proxy set trap
    // So parameters() on the parent should include the ModuleList's children's params
    const params = model.parameters();

    // layers has ScaleModule(4) + BiasModule(4) = 2 params
    // head is Linear(4, 2) = weight + bias = 2 params
    expect(params).toHaveLength(4);
  });

  it("namedParameters() works through nested module hierarchy with ModuleList", () => {
    const model = new ParentModel(api);
    const named = model.namedParameters();

    const names = named.map(([name]) => name);

    // ModuleList children are registered as "0", "1" under "layers"
    expect(names).toContain("layers.0.scale");
    expect(names).toContain("layers.1.bias");
    expect(names).toContain("head.weight");
    expect(names).toContain("head.bias");
  });

  it("assigning Sequential property auto-registers it", () => {
    const model = new SeqParentModel(api);
    const params = model.parameters();

    // body has ScaleModule(4) + BiasModule(4) = 2 params
    expect(params).toHaveLength(2);
  });

  it("namedParameters() works through nested module hierarchy with Sequential", () => {
    const model = new SeqParentModel(api);
    const named = model.namedParameters();

    const names = named.map(([name]) => name);

    expect(names).toContain("body.0.scale");
    expect(names).toContain("body.1.bias");
  });

  it("train()/eval() propagates through auto-registered containers", () => {
    const model = new ParentModel(api);

    model.eval();
    expect(model.training).toBe(false);
    expect(model.layers.training).toBe(false);
    expect(model.layers.get(0).training).toBe(false);
    expect(model.layers.get(1).training).toBe(false);
    expect(model.head.training).toBe(false);

    model.train();
    expect(model.training).toBe(true);
    expect(model.layers.training).toBe(true);
    expect(model.layers.get(0).training).toBe(true);
    expect(model.layers.get(1).training).toBe(true);
    expect(model.head.training).toBe(true);
  });

  it("forward() works through auto-registered hierarchy", async () => {
    const model = new ParentModel(api);
    const input = api.randn([1, 4]);

    const output = model.forward(input);

    // ScaleModule * 1 -> BiasModule + 1 -> Linear [4,2]
    expect(output.shape).toEqual([1, 2]);
  });

  it("stateDict() returns all parameters with correct keys", () => {
    const model = new ParentModel(api);
    const state = model.stateDict();

    expect(Object.keys(state).sort()).toEqual([
      "head.bias",
      "head.weight",
      "layers.0.scale",
      "layers.1.bias",
    ]);
  });
});
