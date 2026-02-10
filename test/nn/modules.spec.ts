/**
 * Tests for nn modules: Linear, LayerNorm, Embedding
 */
import { describe, expect, it, beforeEach } from "vitest";
import { Torchlette } from "../../src/frontend";
import { Linear } from "../../src/nn/linear";
import { LayerNorm } from "../../src/nn/layernorm";
import { Embedding } from "../../src/nn/embedding";

describe("nn.Linear", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette("cpu");
  });

  it("creates weight and bias with correct shapes", () => {
    const linear = new Linear(api, 4, 8);

    expect(linear.inFeatures).toBe(4);
    expect(linear.outFeatures).toBe(8);
    expect(linear.weight.shape).toEqual([8, 4]); // [outFeatures, inFeatures]
    expect(linear.bias).not.toBeNull();
    expect(linear.bias!.shape).toEqual([8]);
  });

  it("creates without bias when bias=false", () => {
    const linear = new Linear(api, 4, 8, { bias: false });

    expect(linear.weight.shape).toEqual([8, 4]);
    expect(linear.bias).toBeNull();
  });

  it("forward produces correct output shape", async () => {
    const linear = new Linear(api, 4, 8);
    const input = api.randn([2, 4]); // batch=2, inFeatures=4

    const output = linear.forward(input);

    expect(output.shape).toEqual([2, 8]); // batch=2, outFeatures=8
  });

  it("forward works with 3D input", async () => {
    const linear = new Linear(api, 4, 8);
    const input = api.randn([2, 3, 4]); // batch=2, seq=3, inFeatures=4

    const output = linear.forward(input);

    expect(output.shape).toEqual([2, 3, 8]); // batch=2, seq=3, outFeatures=8
  });

  it("forward without bias works", async () => {
    const linear = new Linear(api, 4, 8, { bias: false });
    const input = api.randn([2, 4]);

    const output = linear.forward(input);

    expect(output.shape).toEqual([2, 8]);
  });

  it("parameters returns weight and bias", () => {
    const linear = new Linear(api, 4, 8);
    const params = linear.parameters();

    expect(params).toHaveLength(2);
    expect(params[0]).toBe(linear.weight);
    expect(params[1]).toBe(linear.bias);
  });

  it("parameters returns only weight when bias=false", () => {
    const linear = new Linear(api, 4, 8, { bias: false });
    const params = linear.parameters();

    expect(params).toHaveLength(1);
    expect(params[0]).toBe(linear.weight);
  });

  it("weight has requiresGrad=true", () => {
    const linear = new Linear(api, 4, 8);

    expect(linear.weight.requiresGrad).toBe(true);
    expect(linear.bias!.requiresGrad).toBe(true);
  });
});

describe("nn.LayerNorm", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette("cpu");
  });

  it("creates weight and bias with correct shapes", () => {
    const ln = new LayerNorm(api, 8);

    expect(ln.normalizedShape).toBe(8);
    expect(ln.eps).toBe(1e-5);
    expect(ln.weight).not.toBeNull();
    expect(ln.weight!.shape).toEqual([8]);
    expect(ln.bias).not.toBeNull();
    expect(ln.bias!.shape).toEqual([8]);
  });

  it("allows custom eps", () => {
    const ln = new LayerNorm(api, 8, { eps: 1e-6 });

    expect(ln.eps).toBe(1e-6);
  });

  it("forward produces correct output shape", async () => {
    const ln = new LayerNorm(api, 8);
    const input = api.randn([2, 8]); // batch=2, features=8

    const output = ln.forward(input);

    expect(output.shape).toEqual([2, 8]);
  });

  it("forward works with 3D input", async () => {
    const ln = new LayerNorm(api, 8);
    const input = api.randn([2, 3, 8]); // batch=2, seq=3, features=8

    const output = ln.forward(input);

    expect(output.shape).toEqual([2, 3, 8]);
  });

  it("normalizes the last dimension", async () => {
    const ln = new LayerNorm(api, 4);
    // Create input with known values
    const input = api.tensorFromArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);

    const output = ln.forward(input);
    const outputData = await output.cpu();

    // After normalization, each row should have mean~0 and std~1
    // Row 1: [1,2,3,4] -> mean=2.5, std~1.12
    // Row 2: [5,6,7,8] -> mean=6.5, std~1.12
    expect(output.shape).toEqual([2, 4]);

    // Check that output is normalized (rough check)
    const row1 = outputData.slice(0, 4);
    const row1Mean = row1.reduce((a, b) => a + b, 0) / 4;
    expect(Math.abs(row1Mean)).toBeLessThan(0.1); // mean should be close to 0
  });

  it("parameters returns weight and bias", () => {
    const ln = new LayerNorm(api, 8);
    const params = ln.parameters();

    expect(params).toHaveLength(2);
    expect(params[0]).toBe(ln.weight);
    expect(params[1]).toBe(ln.bias);
  });

  it("throws when elementwiseAffine=false", () => {
    const ln = new LayerNorm(api, 8, { elementwiseAffine: false });
    const input = api.randn([2, 8]);

    expect(() => ln.forward(input)).toThrow(
      "LayerNorm without elementwiseAffine is not yet supported",
    );
  });

  it("weight and bias have requiresGrad=true", () => {
    const ln = new LayerNorm(api, 8);

    expect(ln.weight!.requiresGrad).toBe(true);
    expect(ln.bias!.requiresGrad).toBe(true);
  });
});

describe("nn.Embedding", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette("cpu");
  });

  it("creates weight with correct shape", () => {
    const embed = new Embedding(api, 100, 16); // vocab=100, dim=16

    expect(embed.numEmbeddings).toBe(100);
    expect(embed.embeddingDim).toBe(16);
    expect(embed.weight.shape).toEqual([100, 16]);
  });

  it("forward produces correct output shape for 1D input", async () => {
    const embed = new Embedding(api, 100, 16);
    const indices = api.tensorFromArray([0, 5, 10, 15], [4]);

    const output = embed.forward(indices);

    expect(output.shape).toEqual([4, 16]); // [seqLen, embedDim]
  });

  it("forward produces correct output shape for 2D input", async () => {
    const embed = new Embedding(api, 100, 16);
    const indices = api.tensorFromArray([0, 1, 2, 3, 4, 5], [2, 3]); // batch=2, seq=3

    const output = embed.forward(indices);

    expect(output.shape).toEqual([2, 3, 16]); // [batch, seq, embedDim]
  });

  it("looks up correct embeddings", async () => {
    const embed = new Embedding(api, 10, 4);

    // Get the weight values
    const weights = await embed.weight.cpu();

    // Look up indices 0 and 1
    const indices = api.tensorFromArray([0, 1], [2]);
    const output = embed.forward(indices);
    const outputData = await output.cpu();

    // output[0] should equal weight[0]
    for (let i = 0; i < 4; i++) {
      expect(Math.abs(outputData[i] - weights[i])).toBeLessThan(1e-5);
    }

    // output[1] should equal weight[1]
    for (let i = 0; i < 4; i++) {
      expect(Math.abs(outputData[4 + i] - weights[4 + i])).toBeLessThan(1e-5);
    }
  });

  it("parameters returns weight", () => {
    const embed = new Embedding(api, 100, 16);
    const params = embed.parameters();

    expect(params).toHaveLength(1);
    expect(params[0]).toBe(embed.weight);
  });

  it("weight has requiresGrad=true", () => {
    const embed = new Embedding(api, 100, 16);

    expect(embed.weight.requiresGrad).toBe(true);
  });
});
