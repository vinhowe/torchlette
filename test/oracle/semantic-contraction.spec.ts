/**
 * Semantic Derivation — the CONTRACTION adjoint fact (Crystal Campaign 3, §19
 * closure). The framework's last two hand-written adjoints (`matmulBackward`,
 * `linearBackward`) now DERIVE from ONE fact: the adjoint of a contraction is two
 * contractions with transpose flags flipped per a fixed rule
 * (src/ops/semantic/contraction.ts). This gate proves:
 *
 *  (1) THE FACT is correct as pure DATA — `contractionAdjoint` returns the exact
 *      transpose-flag structure for all four (ta,tb) combos, and is a DATA term
 *      (schema gate). Independently transcribed here, so it is a two-source diff.
 *  (2) THE REALIZER is a theorem — for every (ta,tb), 2-D and batched, the
 *      realized grads match torch autograd of the same contraction (the
 *      `contraction_backward` oracle referee).
 *  (3) THE LANDED PATH is correct — the frontend `matmul` (the (F,F) row) and
 *      `linear` (the (F,T) row + bias) grads match torch, in f32 AND autocast-f16,
 *      2-D and batched, with and without bias.
 */

import { describe, expect, test } from "vitest";

import { Torchlette } from "../../src";
import {
  type AdjointContraction,
  assertNoContractionBody,
  type Contraction,
  contractionAdjoint,
  realizeContractionAdjoint,
} from "../../src/ops/semantic/contraction";
import { assertClose } from "../helpers/assertions";
import { runTorchOracleBackwardBatch } from "./torch-oracle";

// ---- seeded data ----------------------------------------------------------
function rand(n: number, seed: number): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out.push((s / 0xffffffff) * 2 - 1);
  }
  return out;
}
const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

// ===========================================================================
// (1) THE FACT — pure DATA. An INDEPENDENT transcription of the four rows.
// ===========================================================================

describe("contraction fact — the adjoint is two contractions (DATA)", () => {
  // Independent transcription (module header): C = op(A,ta)·op(B,tb) ⇒
  //   (F,F): dA = G·Bᵀ , dB = Aᵀ·G      (F,T): dA = G·B , dB = Gᵀ·A
  //   (T,F): dA = B·Gᵀ , dB = A·G       (T,T): dA = Bᵀ·Gᵀ, dB = Gᵀ·Aᵀ
  const EXPECT: Record<
    string,
    { dA: AdjointContraction; dB: AdjointContraction }
  > = {
    "false,false": {
      dA: { lhs: "G", rhs: "B", ta: false, tb: true },
      dB: { lhs: "A", rhs: "G", ta: true, tb: false },
    },
    "false,true": {
      dA: { lhs: "G", rhs: "B", ta: false, tb: false },
      dB: { lhs: "G", rhs: "A", ta: true, tb: false },
    },
    "true,false": {
      dA: { lhs: "B", rhs: "G", ta: false, tb: true },
      dB: { lhs: "A", rhs: "G", ta: false, tb: false },
    },
    "true,true": {
      dA: { lhs: "B", rhs: "G", ta: true, tb: true },
      dB: { lhs: "G", rhs: "A", ta: true, tb: true },
    },
  };

  for (const ta of [false, true])
    for (const tb of [false, true]) {
      test(`ta=${ta}, tb=${tb} descriptors`, () => {
        const got = contractionAdjoint({ ta, tb });
        expect(got).toEqual(EXPECT[`${ta},${tb}`]);
        expect(() => assertNoContractionBody(got.dA)).not.toThrow();
        expect(() => assertNoContractionBody(got.dB)).not.toThrow();
      });
    }

  test("schema gate rejects a smuggled body leaf", () => {
    expect(() =>
      assertNoContractionBody({
        lhs: "A",
        rhs: "G",
        // biome-ignore lint: intentional bad leaf for the negative test
        ta: (() => true) as never,
        tb: false,
      }),
    ).toThrow(/schema gate/);
  });
});

// ===========================================================================
// (2) THE REALIZER — a theorem, refereed by torch for all four flag combos.
// ===========================================================================

describe("contraction realizer — torch-refereed for all flag combos", () => {
  const api = new Torchlette("cpu");
  // Square inner dims so every (ta,tb) is shape-legal: A=[.,S,S], B=[.,S,S].
  const S = 4;
  const CASES: { batch: number[]; label: string }[] = [
    { batch: [], label: "2D" },
    { batch: [3], label: "batched" },
  ];

  for (const { batch, label } of CASES)
    for (const ta of [false, true])
      for (const tb of [false, true]) {
        test(`${label} ta=${ta} tb=${tb}`, async () => {
          const aShape = [...batch, S, S];
          const bShape = [...batch, S, S];
          const cShape = [...batch, S, S]; // op(A,ta)·op(B,tb) is [.,S,S]
          const aV = rand(prod(aShape), 11 + (ta ? 1 : 0));
          const bV = rand(prod(bShape), 37 + (tb ? 1 : 0));
          const gV = rand(prod(cShape), 91 + batch.length);

          const A = api.tensorFromArray(aV, aShape)._unwrap();
          const B = api.tensorFromArray(bV, bShape)._unwrap();
          const G = api.tensorFromArray(gV, cShape)._unwrap();

          const fwd: Contraction = { ta, tb };
          const { dA, dB } = contractionAdjoint(fwd);
          const ctx = { rt: api.runtime, A, B, G };
          const gradA = realizeContractionAdjoint(ctx, dA);
          const gradB = realizeContractionAdjoint(ctx, dB);
          const gA = await api.runtime.cpu(gradA);
          const gB = await api.runtime.cpu(gradB);

          const [oracle] = await runTorchOracleBackwardBatch([
            {
              op: "contraction_backward",
              caseName: `contraction.${label}.${ta}.${tb}`,
              inputs: [
                { values: aV, shape: aShape },
                { values: bV, shape: bShape },
                { values: gV, shape: cShape },
              ],
              options: { ta, tb },
            },
          ]);

          assertClose({ shape: gradA.shape, values: gA }, oracle.grads[0], 1e-4, 1e-3, "dA");
          assertClose({ shape: gradB.shape, values: gB }, oracle.grads[1], 1e-4, 1e-3, "dB");
        });
      }
});

// ===========================================================================
// (3) THE LANDED PATH — frontend matmul/linear grads match torch.
// ===========================================================================

async function readGrad(t: {
  grad: { shape: number[]; cpu(): Promise<number[]> } | null;
}) {
  if (!t.grad) throw new Error("missing grad");
  return { shape: t.grad.shape, values: await t.grad.cpu() };
}

describe("landed matmul — the (F,F) contraction row", () => {
  for (const [batch, label] of [
    [[], "2D"],
    [[2], "batched"],
  ] as [number[], string][]) {
    for (const amp of [false, true]) {
      test(`${label}${amp ? " autocast-f16" : ""}`, async () => {
        const api = new Torchlette("cpu");
        const aShape = [...batch, 3, 5];
        const bShape = [...batch, 5, 4];
        const aV = rand(prod(aShape), 5);
        const bV = rand(prod(bShape), 6);
        const a = api.tensorFromArray(aV, aShape, { requiresGrad: true });
        const b = api.tensorFromArray(bV, bShape, { requiresGrad: true });
        const c = amp ? api.autocast(() => a.matmul(b)) : a.matmul(b);
        const loss = c.sum();
        const scalar = typeof loss === "number" ? null : loss.reshape([]);
        if (!scalar) throw new Error("expected tensor loss");
        await scalar.backward();

        const [oracle] = await runTorchOracleBackwardBatch([
          {
            op: "backward",
            caseName: `matmul.${label}`,
            inputs: [
              { values: aV, shape: aShape },
              { values: bV, shape: bShape },
            ],
            options: { op: "matmul", requiresGrad: [true, true] },
          },
        ]);

        const tol = amp ? 5e-2 : 1e-4;
        assertClose(await readGrad(a), oracle.grads[0], tol, tol, "dA");
        assertClose(await readGrad(b), oracle.grads[1], tol, tol, "dB");
      });
    }
  }
});

describe("landed linear — the (F,T) contraction row + bias", () => {
  for (const [batch, label] of [
    [[], "2D"],
    [[2], "batched"],
  ] as [number[], string][]) {
    for (const bias of [false, true]) {
      for (const amp of [false, true]) {
        test(`${label}${bias ? " +bias" : ""}${amp ? " autocast-f16" : ""}`, async () => {
          const api = new Torchlette("cpu");
          const inF = 5;
          const outF = 4;
          const xShape = [...batch, 3, inF];
          const wShape = [outF, inF];
          const xV = rand(prod(xShape), 15);
          const wV = rand(prod(wShape), 16);
          const bV = rand(outF, 17);
          const x = api.tensorFromArray(xV, xShape, { requiresGrad: true });
          const w = api.tensorFromArray(wV, wShape, { requiresGrad: true });
          const bT = bias
            ? api.tensorFromArray(bV, [outF], { requiresGrad: true })
            : null;
          const y = amp
            ? api.autocast(() => api.linear(x, w, bT))
            : api.linear(x, w, bT);
          const loss = y.sum();
          const scalar = typeof loss === "number" ? null : loss.reshape([]);
          if (!scalar) throw new Error("expected tensor loss");
          await scalar.backward();

          const inputs = [
            { values: xV, shape: xShape },
            { values: wV, shape: wShape },
          ];
          if (bias) inputs.push({ values: bV, shape: [outF] });
          const [oracle] = await runTorchOracleBackwardBatch([
            {
              op: "linear_backward",
              caseName: `linear.${label}`,
              inputs,
            },
          ]);

          const tol = amp ? 5e-2 : 1e-4;
          assertClose(await readGrad(x), oracle.grads[0], tol, tol, "dX");
          assertClose(await readGrad(w), oracle.grads[1], tol, tol, "dW");
          if (bias && bT)
            assertClose(await readGrad(bT), oracle.grads[2], tol, tol, "dBias");
        });
      }
    }
  }
});
