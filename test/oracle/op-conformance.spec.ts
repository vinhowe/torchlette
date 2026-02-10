import fc from "fast-check";
import { describe, expect, test } from "vitest";

import { OP_SPECS } from "../../ops/specs/registry";
import type { OpCase, OpSpec } from "../../ops/specs/types";
import {
  runtimeAdd,
  runtimeCpu,
  runtimeGather,
  runtimeMatmul,
  runtimeMean,
  runtimeMul,
  runtimeRelu,
  runtimeReshape,
  runtimeScatterAdd,
  runtimeSqrt,
  runtimeSub,
  runtimeSum,
  runtimeTensorFromArray,
  runtimeTranspose,
} from "../../src";
import { runTorchOracleBatch } from "./torch-oracle";

type OutputPayload = { shape: number[]; values: number[] };

const DEFAULT_ATOL = 1e-5;
const DEFAULT_RTOL = 1e-4;
const PROPERTY_RUNS_MATCH = 3;
const PROPERTY_RUNS_FAIL = 1;
const FAIL_ASSERT_OPTIONS = { numRuns: PROPERTY_RUNS_FAIL, endOnFailure: true };
const ORACLE_TIMEOUT = 60000;
const oracleTest = (name: string, fn: () => Promise<void> | void): void => {
  test(name, fn, ORACLE_TIMEOUT);
};

const dimArb = fc.integer({ min: 1, max: 4 });
const shapeArb = fc.array(dimArb, { minLength: 1, maxLength: 3 });
const shape2dArb = fc.array(dimArb, { minLength: 2, maxLength: 3 });
const scalarArb = fc.float({
  min: -5,
  max: 5,
  noNaN: true,
  noDefaultInfinity: true,
});
const sqrtScalarArb = fc.float({
  min: 0,
  max: 5,
  noNaN: true,
  noDefaultInfinity: true,
});

const sizeOf = (shape: number[]): number =>
  shape.reduce((acc, dim) => acc * dim, 1);

const normalizeDim = (dim: number, rank: number): number =>
  dim < 0 ? rank + dim : dim;

const tensorArb = shapeArb.chain((shape) => {
  const size = sizeOf(shape);
  return fc
    .array(scalarArb, { minLength: size, maxLength: size })
    .map((values) => ({ shape, values }));
});
const sqrtTensorArb = shapeArb.chain((shape) => {
  const size = sizeOf(shape);
  return fc
    .array(sqrtScalarArb, { minLength: size, maxLength: size })
    .map((values) => ({ shape, values }));
});
const gatherCaseArb = shapeArb.chain((inputShape) => {
  const inputSize = sizeOf(inputShape);
  const rank = inputShape.length;
  return fc.integer({ min: -rank, max: rank - 1 }).chain((dim) => {
    const dimNorm = normalizeDim(dim, rank);
    const indexShapeArb = fc
      .tuple(
        ...inputShape.map((size, axis) => {
          const max = axis === dimNorm ? Math.max(1, size * 2) : size;
          return fc.integer({ min: 1, max });
        }),
      )
      .map((values) => Array.from(values));
    const inputValuesArb = fc.array(scalarArb, {
      minLength: inputSize,
      maxLength: inputSize,
    });

    return fc
      .tuple(indexShapeArb, inputValuesArb)
      .chain(([indexShape, values]) => {
        const indexSize = sizeOf(indexShape);
        const indexValuesArb = fc.array(
          fc.integer({ min: 0, max: inputShape[dimNorm] - 1 }),
          { minLength: indexSize, maxLength: indexSize },
        );
        return indexValuesArb.map((indices) => ({
          inputShape,
          indexShape,
          values,
          indices,
          dim,
        }));
      });
  });
});
const scatterAddCaseArb = shapeArb.chain((inputShape) => {
  const inputSize = sizeOf(inputShape);
  const rank = inputShape.length;
  return fc.integer({ min: -rank, max: rank - 1 }).chain((dim) => {
    const dimNorm = normalizeDim(dim, rank);
    const indexShapeArb = fc
      .tuple(
        ...inputShape.map((size, axis) => {
          const max = axis === dimNorm ? Math.max(1, size * 2) : size;
          return fc.integer({ min: 1, max });
        }),
      )
      .map((values) => Array.from(values));
    const inputValuesArb = fc.array(scalarArb, {
      minLength: inputSize,
      maxLength: inputSize,
    });

    return fc
      .tuple(indexShapeArb, inputValuesArb)
      .chain(([indexShape, input]) => {
        const indexSize = sizeOf(indexShape);
        const srcValuesArb = fc.array(scalarArb, {
          minLength: indexSize,
          maxLength: indexSize,
        });
        const indexValuesArb = fc.array(
          fc.integer({ min: 0, max: inputShape[dimNorm] - 1 }),
          { minLength: indexSize, maxLength: indexSize },
        );
        return fc.tuple(srcValuesArb, indexValuesArb).map(([src, indices]) => ({
          inputShape,
          indexShape,
          input,
          src,
          indices,
          dim,
        }));
      });
  });
});
const gatherNegativeIndexCaseArb = gatherCaseArb.map((item) => ({
  ...item,
  indices: item.indices.map((value, index) => (index === 0 ? -1 : value)),
}));
const scatterAddNegativeIndexCaseArb = scatterAddCaseArb.map((item) => ({
  ...item,
  indices: item.indices.map((value, index) => (index === 0 ? -1 : value)),
}));

const sumDimCaseArb = shapeArb.chain((shape) => {
  const size = sizeOf(shape);
  const rank = shape.length;
  const dims = Array.from({ length: rank }, (_, index) => index);
  return fc
    .record({
      values: fc.array(scalarArb, { minLength: size, maxLength: size }),
      dims: fc.subarray(dims, {
        minLength: 1,
        maxLength: Math.min(2, rank),
      }),
      keepdim: fc.boolean(),
    })
    .map(({ values, dims, keepdim }) => ({
      shape,
      values,
      dims,
      keepdim,
    }));
});

const binaryCaseArb = tensorArb.chain(({ shape, values }) => {
  const size = values.length;
  return fc
    .array(scalarArb, { minLength: size, maxLength: size })
    .map((valuesB) => ({ shape, valuesA: values, valuesB }));
});

const broadcastShapeArb = fc.array(dimArb, { minLength: 0, maxLength: 3 });
const inputShapeArb = (outShape: number[]) => {
  const outRank = outShape.length;
  return fc.integer({ min: 0, max: outRank }).chain((drop) => {
    const tail = outShape.slice(drop);
    return fc
      .array(fc.boolean(), { minLength: tail.length, maxLength: tail.length })
      .map((mask) => tail.map((dim, idx) => (mask[idx] ? dim : 1)));
  });
};
const broadcastBinaryCaseArb = broadcastShapeArb.chain((outShape) => {
  if (outShape.length === 0) {
    return fc.record({
      shapeA: fc.constant([]),
      shapeB: fc.constant([]),
      valuesA: fc.array(scalarArb, { minLength: 1, maxLength: 1 }),
      valuesB: fc.array(scalarArb, { minLength: 1, maxLength: 1 }),
    });
  }
  return fc
    .tuple(inputShapeArb(outShape), inputShapeArb(outShape))
    .chain(([shapeA, shapeB]) => {
      const sizeA = sizeOf(shapeA);
      const sizeB = sizeOf(shapeB);
      return fc
        .tuple(
          fc.array(scalarArb, { minLength: sizeA, maxLength: sizeA }),
          fc.array(scalarArb, { minLength: sizeB, maxLength: sizeB }),
        )
        .map(([valuesA, valuesB]) => ({
          shapeA,
          shapeB,
          valuesA,
          valuesB,
        }));
    });
});

const matmulCaseArb = fc.tuple(dimArb, dimArb, dimArb).chain(([m, k, n]) => {
  const sizeA = m * k;
  const sizeB = k * n;
  return fc
    .tuple(
      fc.array(scalarArb, { minLength: sizeA, maxLength: sizeA }),
      fc.array(scalarArb, { minLength: sizeB, maxLength: sizeB }),
    )
    .map(([valuesA, valuesB]) => ({
      shapeA: [m, k],
      shapeB: [k, n],
      valuesA,
      valuesB,
    }));
});

const transposeCaseArb = shape2dArb.chain((shape) => {
  const size = shape.reduce((acc, dim) => acc * dim, 1);
  return fc
    .tuple(
      fc.array(scalarArb, { minLength: size, maxLength: size }),
      fc
        .tuple(
          fc.integer({ min: 0, max: shape.length - 1 }),
          fc.integer({ min: 0, max: shape.length - 1 }),
        )
        .filter(([dim0, dim1]) => dim0 !== dim1),
    )
    .map(([values, [dim0, dim1]]) => ({
      shape,
      values,
      dim0,
      dim1,
    }));
});

function assertClose(
  actual: OutputPayload,
  expected: OutputPayload,
  options?: { atol?: number; rtol?: number },
) {
  const atol = options?.atol ?? DEFAULT_ATOL;
  const rtol = options?.rtol ?? DEFAULT_RTOL;

  expect(actual.shape).toEqual(expected.shape);
  expect(actual.values.length).toBe(expected.values.length);

  for (let i = 0; i < actual.values.length; i += 1) {
    const a = actual.values[i];
    const b = expected.values[i];
    const diff = Math.abs(a - b);
    const tol = atol + rtol * Math.abs(b);
    expect(diff).toBeLessThanOrEqual(tol);
  }
}

function toPayload(values: number[], shape: number[]): OutputPayload {
  return { values, shape };
}

async function runTorchletteCase(
  spec: OpSpec,
  opCase: OpCase,
): Promise<OutputPayload> {
  const inputs = opCase.inputs.map((input) =>
    runtimeTensorFromArray(input.values, input.shape),
  );

  if (spec.name === "add") {
    if (opCase.options) {
      throw new Error("add options are not implemented yet");
    }
    const out = runtimeAdd(inputs[0], inputs[1]);
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "sub") {
    const options = opCase.options as { alpha?: number } | undefined;
    const out = runtimeSub(inputs[0], inputs[1], options);
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "mul") {
    if (opCase.options) {
      throw new Error("mul options are not implemented yet");
    }
    const out = runtimeMul(inputs[0], inputs[1]);
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "matmul") {
    if (opCase.options) {
      throw new Error("matmul options are not implemented yet");
    }
    const out = runtimeMatmul(inputs[0], inputs[1]);
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "sum") {
    const out = runtimeSum(inputs[0], opCase.options);
    if (typeof out === "number") {
      return toPayload([out], []);
    }
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "sqrt") {
    if (opCase.options) {
      throw new Error("sqrt options are not implemented yet");
    }
    const out = runtimeSqrt(inputs[0]);
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "relu") {
    if (opCase.options) {
      throw new Error("relu options are not implemented yet");
    }
    const out = runtimeRelu(inputs[0]);
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "gather") {
    const options = opCase.options as { dim?: number } | undefined;
    if (!options || options.dim == null) {
      throw new Error("gather requires options.dim");
    }
    const out = runtimeGather(inputs[0], inputs[1], { dim: options.dim });
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "scatterAdd") {
    const options = opCase.options as { dim?: number } | undefined;
    if (!options || options.dim == null) {
      throw new Error("scatterAdd requires options.dim");
    }
    const out = runtimeScatterAdd(inputs[0], inputs[1], inputs[2], {
      dim: options.dim,
    });
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "mean") {
    const out = runtimeMean(inputs[0], opCase.options);
    if (typeof out === "number") {
      return toPayload([out], []);
    }
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "reshape") {
    const options = opCase.options as { shape?: number[] } | undefined;
    if (!options?.shape) {
      throw new Error("reshape requires options.shape");
    }
    const out = runtimeReshape(inputs[0], options.shape);
    return toPayload(await runtimeCpu(out), out.shape);
  }

  if (spec.name === "transpose") {
    const options = opCase.options as
      | { dim0?: number; dim1?: number }
      | undefined;
    if (options?.dim0 == null || options.dim1 == null) {
      throw new Error("transpose requires options.dim0 and options.dim1");
    }
    const out = runtimeTranspose(inputs[0], {
      dim0: options.dim0,
      dim1: options.dim1,
    });
    return toPayload(await runtimeCpu(out), out.shape);
  }

  throw new Error(`Torchlette op not implemented: ${spec.name}`);
}

async function runCases(spec: OpSpec, cases: OpCase[]) {
  const oracleCases = cases.map((opCase) => ({
    op: spec.torchOp,
    caseName: opCase.name,
    inputs: opCase.inputs,
    options: opCase.options,
  }));

  const oracleOutputs = await runTorchOracleBatch(oracleCases);

  for (let i = 0; i < cases.length; i += 1) {
    const opCase = cases[i];
    const expected = oracleOutputs[i];
    const actual = await runTorchletteCase(spec, opCase);
    assertClose(actual, expected, { atol: opCase.atol, rtol: opCase.rtol });
  }
}

async function expectOracleFailure(spec: OpSpec, opCase: OpCase) {
  await expect(
    runTorchOracleBatch([
      {
        op: spec.torchOp,
        caseName: opCase.name,
        inputs: opCase.inputs,
        options: opCase.options,
      },
    ]),
  ).rejects.toThrow();
}

function getSpec(name: string): OpSpec {
  const spec = OP_SPECS.find((candidate) => candidate.name === name);
  if (!spec) {
    throw new Error(`Missing OpSpec: ${name}`);
  }
  return spec;
}

describe("oracle ring-3: PyTorch parity (cpu)", () => {
  oracleTest("add matches PyTorch for random shapes", async () => {
    const spec = getSpec("add");
    await fc.assert(
      fc.asyncProperty(
        fc.array(binaryCaseArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `add.prop.${index}`,
            inputs: [
              { values: item.valuesA, shape: item.shape },
              { values: item.valuesB, shape: item.shape },
            ],
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("add matches PyTorch for broadcastable shapes", async () => {
    const spec = getSpec("add");
    await fc.assert(
      fc.asyncProperty(
        fc.array(broadcastBinaryCaseArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `add.broadcast.prop.${index}`,
            inputs: [
              { values: item.valuesA, shape: item.shapeA },
              { values: item.valuesB, shape: item.shapeB },
            ],
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("mul matches PyTorch for random shapes", async () => {
    const spec = getSpec("mul");
    await fc.assert(
      fc.asyncProperty(
        fc.array(binaryCaseArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `mul.prop.${index}`,
            inputs: [
              { values: item.valuesA, shape: item.shape },
              { values: item.valuesB, shape: item.shape },
            ],
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("mul matches PyTorch for broadcastable shapes", async () => {
    const spec = getSpec("mul");
    await fc.assert(
      fc.asyncProperty(
        fc.array(broadcastBinaryCaseArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `mul.broadcast.prop.${index}`,
            inputs: [
              { values: item.valuesA, shape: item.shapeA },
              { values: item.valuesB, shape: item.shapeB },
            ],
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("matmul matches PyTorch for random shapes", async () => {
    const spec = getSpec("matmul");
    await fc.assert(
      fc.asyncProperty(
        fc.array(matmulCaseArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `matmul.prop.${index}`,
            inputs: [
              { values: item.valuesA, shape: item.shapeA },
              { values: item.valuesB, shape: item.shapeB },
            ],
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("matmul matches PyTorch for 1D and broadcast cases", async () => {
    const spec = getSpec("matmul");
    const opCases: OpCase[] = [
      {
        name: "matmul.1d.1d",
        inputs: [
          { values: [1, 2, 3], shape: [3] },
          { values: [4, 5, 6], shape: [3] },
        ],
        expectation: "match",
      },
      {
        name: "matmul.2d.1d",
        inputs: [
          { values: [1, 2, 3, 4, 5, 6], shape: [2, 3] },
          { values: [7, 8, 9], shape: [3] },
        ],
        expectation: "match",
      },
      {
        name: "matmul.1d.2d",
        inputs: [
          { values: [1, 2, 3], shape: [3] },
          { values: [4, 5, 6, 7, 8, 9], shape: [3, 2] },
        ],
        expectation: "match",
      },
      {
        name: "matmul.broadcast.batch",
        inputs: [
          { values: [1, 0, 0, 1], shape: [1, 2, 2] },
          {
            values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            shape: [3, 2, 2],
          },
        ],
        expectation: "match",
      },
    ];
    await runCases(spec, opCases);
  });

  oracleTest("sum matches PyTorch for random shapes", async () => {
    const spec = getSpec("sum");
    await fc.assert(
      fc.asyncProperty(
        fc.array(tensorArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `sum.prop.${index}`,
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest(
    "sum with dim/keepdim matches PyTorch for random shapes",
    async () => {
      const spec = getSpec("sum");
      await fc.assert(
        fc.asyncProperty(
          fc.array(sumDimCaseArb, { minLength: 1, maxLength: 3 }),
          async (cases) => {
            const opCases = cases.map((item, index) => ({
              name: `sum.dim.prop.${index}`,
              inputs: [{ values: item.values, shape: item.shape }],
              options: {
                dim: item.dims.length === 1 ? item.dims[0] : item.dims,
                keepdim: item.keepdim,
              },
              expectation: "match",
            }));
            await runCases(spec, opCases);
          },
        ),
        { numRuns: PROPERTY_RUNS_MATCH },
      );
    },
  );

  oracleTest("sqrt matches PyTorch for random shapes", async () => {
    const spec = getSpec("sqrt");
    await fc.assert(
      fc.asyncProperty(
        fc.array(sqrtTensorArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `sqrt.prop.${index}`,
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("gather matches PyTorch for random shapes", async () => {
    const spec = getSpec("gather");
    await fc.assert(
      fc.asyncProperty(
        fc.array(gatherCaseArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `gather.prop.${index}`,
            inputs: [
              { values: item.values, shape: item.inputShape },
              { values: item.indices, shape: item.indexShape, dtype: "i32" },
            ],
            options: { dim: item.dim },
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("scatterAdd matches PyTorch for random shapes", async () => {
    const spec = getSpec("scatterAdd");
    await fc.assert(
      fc.asyncProperty(
        fc.array(scatterAddCaseArb, { minLength: 1, maxLength: 3 }),
        async (cases) => {
          const opCases = cases.map((item, index) => ({
            name: `scatterAdd.prop.${index}`,
            inputs: [
              { values: item.input, shape: item.inputShape },
              { values: item.indices, shape: item.indexShape, dtype: "i32" },
              { values: item.src, shape: item.indexShape },
            ],
            options: { dim: item.dim },
            expectation: "match",
          }));
          await runCases(spec, opCases);
        },
      ),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("gather rejects negative indices", async () => {
    const spec = getSpec("gather");
    await fc.assert(
      fc.asyncProperty(gatherNegativeIndexCaseArb, async (item) => {
        const opCase = {
          name: "gather.neg",
          inputs: [
            { values: item.values, shape: item.inputShape },
            { values: item.indices, shape: item.indexShape, dtype: "i32" },
          ],
          options: { dim: item.dim },
          expectation: "match",
        };
        await expectOracleFailure(spec, opCase);
        await expect(runTorchletteCase(spec, opCase)).rejects.toThrow();
      }),
      { numRuns: PROPERTY_RUNS_FAIL },
    );
  });

  oracleTest("scatterAdd rejects negative indices", async () => {
    const spec = getSpec("scatterAdd");
    await fc.assert(
      fc.asyncProperty(scatterAddNegativeIndexCaseArb, async (item) => {
        const opCase = {
          name: "scatterAdd.neg",
          inputs: [
            { values: item.input, shape: item.inputShape },
            { values: item.indices, shape: item.indexShape, dtype: "i32" },
            { values: item.src, shape: item.indexShape },
          ],
          options: { dim: item.dim },
          expectation: "match",
        };
        await expectOracleFailure(spec, opCase);
        await expect(runTorchletteCase(spec, opCase)).rejects.toThrow();
      }),
      { numRuns: PROPERTY_RUNS_FAIL },
    );
  });

  oracleTest("sub matches PyTorch for random shapes", async () => {
    const spec = getSpec("sub");
    await fc.assert(
      fc.asyncProperty(binaryCaseArb, async (item) => {
        const opCases = [
          {
            name: "sub.prop.0",
            inputs: [
              { values: item.valuesA, shape: item.shape },
              { values: item.valuesB, shape: item.shape },
            ],
            expectation: "match",
          },
        ];
        await runCases(spec, opCases);
      }),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  test.fails("div planned cases fail for random shapes", async () => {
    const spec = getSpec("div");
    await fc.assert(
      fc.asyncProperty(binaryCaseArb, async (item) => {
        const opCases = [
          {
            name: "div.prop.0",
            inputs: [
              { values: item.valuesA, shape: item.shape },
              { values: item.valuesB, shape: item.shape },
            ],
            expectation: "expected_failure",
          },
        ];
        await runCases(spec, opCases);
      }),
      FAIL_ASSERT_OPTIONS,
    );
  });

  test.fails("neg planned cases fail for random shapes", async () => {
    const spec = getSpec("neg");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const opCases = [
          {
            name: "neg.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "expected_failure",
          },
        ];
        await runCases(spec, opCases);
      }),
      FAIL_ASSERT_OPTIONS,
    );
  });

  test.fails("abs planned cases fail for random shapes", async () => {
    const spec = getSpec("abs");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const opCases = [
          {
            name: "abs.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "expected_failure",
          },
        ];
        await runCases(spec, opCases);
      }),
      FAIL_ASSERT_OPTIONS,
    );
  });

  test.fails("exp planned cases fail for random shapes", async () => {
    const spec = getSpec("exp");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const opCases = [
          {
            name: "exp.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "expected_failure",
          },
        ];
        await runCases(spec, opCases);
      }),
      FAIL_ASSERT_OPTIONS,
    );
  });

  test.fails("log planned cases fail for random shapes", async () => {
    const spec = getSpec("log");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const opCases = [
          {
            name: "log.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "expected_failure",
          },
        ];
        await runCases(spec, opCases);
      }),
      FAIL_ASSERT_OPTIONS,
    );
  });

  oracleTest("relu matches PyTorch for random shapes", async () => {
    const spec = getSpec("relu");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const opCases = [
          {
            name: "relu.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "match",
          },
        ];
        await runCases(spec, opCases);
      }),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("mean matches PyTorch for random shapes", async () => {
    const spec = getSpec("mean");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const opCases = [
          {
            name: "mean.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "match",
          },
        ];
        await runCases(spec, opCases);
      }),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("reshape matches PyTorch for random shapes", async () => {
    const spec = getSpec("reshape");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const size = item.shape.reduce((acc, dim) => acc * dim, 1);
        const opCases = [
          {
            name: "reshape.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            options: { shape: [size] },
            expectation: "match",
          },
        ];
        await runCases(spec, opCases);
      }),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  oracleTest("transpose matches PyTorch for random shapes", async () => {
    const spec = getSpec("transpose");
    await fc.assert(
      fc.asyncProperty(transposeCaseArb, async (item) => {
        const opCases = [
          {
            name: "transpose.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            options: { dim0: item.dim0, dim1: item.dim1 },
            expectation: "match",
          },
        ];
        await runCases(spec, opCases);
      }),
      { numRuns: PROPERTY_RUNS_MATCH },
    );
  });

  test.fails("linalg.norm planned cases fail for random shapes", async () => {
    const spec = getSpec("linalg.norm");
    await fc.assert(
      fc.asyncProperty(tensorArb, async (item) => {
        const opCases = [
          {
            name: "linalg.norm.prop.0",
            inputs: [{ values: item.values, shape: item.shape }],
            expectation: "expected_failure",
          },
        ];
        await runCases(spec, opCases);
      }),
      FAIL_ASSERT_OPTIONS,
    );
  });
});
