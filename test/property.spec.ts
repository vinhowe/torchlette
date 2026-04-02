import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { add, mul, sum, tensorFromArray } from "../src/backend/cpu";

describe("property tests: numeric small shapes", () => {
  const smallTensorArb = fc
    .array(fc.integer({ min: -3, max: 3 }), { minLength: 1, maxLength: 9 })
    .map((values) => {
      const shape = [values.length];
      return tensorFromArray(values, shape);
    });

  it("add and mul are commutative for small tensors", () => {
    fc.assert(
      fc.property(smallTensorArb, smallTensorArb, (a, b) => {
        fc.pre(a.size === b.size);
        const add1 = add(a, b).toArray();
        const add2 = add(b, a).toArray();
        const mul1 = mul(a, b).toArray();
        const mul2 = mul(b, a).toArray();
        expect(add1).toEqual(add2);
        expect(mul1).toEqual(mul2);
      }),
      { numRuns: 60 },
    );
  });

  it("sum is additive over elementwise add", () => {
    fc.assert(
      fc.property(smallTensorArb, smallTensorArb, (a, b) => {
        fc.pre(a.size === b.size);
        // sum returns 0-d tensor, extract value with toArray()[0]
        const sumAdd = sum(add(a, b)).toArray()[0];
        const sumA = sum(a).toArray()[0];
        const sumB = sum(b).toArray()[0];
        expect(sumAdd).toBeCloseTo(sumA + sumB, 5);
      }),
      { numRuns: 60 },
    );
  });
});
