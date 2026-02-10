import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { add, Engine, mul, sum, tensorFromArray } from "../src";

type RawOp =
  | { kind: "scheduleLoc"; locId: number }
  | { kind: "commitLoc"; locId: number }
  | { kind: "baseCommit"; baseId: number }
  | { kind: "rngDraw"; opNonce: number };

type Op =
  | { kind: "scheduleLoc"; locId: number }
  | { kind: "commitLoc"; locId: number }
  | { kind: "baseCommit"; baseId: number; mutId: number }
  | { kind: "rngDraw"; opNonce: number };

const locIdArb = fc.integer({ min: 1, max: 4 });
const baseIdArb = fc.integer({ min: 1, max: 4 });
const opNonceArb = fc.integer({ min: 1, max: 4 });

const rawOpsArb = fc.array(
  fc.oneof(
    locIdArb.map((locId) => ({ kind: "scheduleLoc", locId })),
    locIdArb.map((locId) => ({ kind: "commitLoc", locId })),
    baseIdArb.map((baseId) => ({ kind: "baseCommit", baseId })),
    opNonceArb.map((opNonce) => ({ kind: "rngDraw", opNonce })),
  ),
  { minLength: 1, maxLength: 25 },
);

function normalizeOps(ops: RawOp[]): Op[] {
  let nextMutId = 1;
  return ops.map((op) => {
    if (op.kind === "baseCommit") {
      return { ...op, mutId: nextMutId++ };
    }
    return op;
  });
}

function applyOps(engine: Engine, ops: Op[]): void {
  engine._debug_setRngBasis({ algorithmId: 0, seed: 0 });
  for (const op of ops) {
    if (op.kind === "scheduleLoc") {
      engine._debug_scheduleLocAccess(op.locId);
    } else if (op.kind === "commitLoc") {
      engine._debug_commitLocStore(op.locId);
    } else if (op.kind === "baseCommit") {
      engine._debug_baseCommit(op.baseId, op.mutId);
    } else {
      engine._debug_random(op.opNonce);
    }
  }
}

function countOps(ops: Op[]): {
  locLogical: Record<string, number>;
  locCommit: Record<string, number>;
  baseCommit: Record<string, number>;
  baseMutIds: Record<string, number[]>;
  rngDraws: Record<string, number[]>;
} {
  const locLogical: Record<string, number> = {};
  const locCommit: Record<string, number> = {};
  const baseCommit: Record<string, number> = {};
  const baseMutIds: Record<string, number[]> = {};
  const rngDraws: Record<string, number[]> = {};

  for (const op of ops) {
    if (op.kind === "scheduleLoc") {
      const key = op.locId.toString();
      locLogical[key] = (locLogical[key] ?? 0) + 1;
    } else if (op.kind === "commitLoc") {
      const key = op.locId.toString();
      locCommit[key] = (locCommit[key] ?? 0) + 1;
    } else if (op.kind === "baseCommit") {
      const key = op.baseId.toString();
      baseCommit[key] = (baseCommit[key] ?? 0) + 1;
      if (!baseMutIds[key]) {
        baseMutIds[key] = [];
      }
      baseMutIds[key].push(op.mutId);
    } else {
      const key = op.opNonce.toString();
      if (!rngDraws[key]) {
        rngDraws[key] = [];
      }
      rngDraws[key].push(op.opNonce);
    }
  }

  return { locLogical, locCommit, baseCommit, baseMutIds, rngDraws };
}

describe("property tests: determinism and monotonicity", () => {
  it("builds identical plans for equivalent programs", () => {
    fc.assert(
      fc.property(rawOpsArb, (rawOps) => {
        const ops = normalizeOps(rawOps);
        const engineA = new Engine();
        const engineB = new Engine();

        applyOps(engineA, ops);
        applyOps(engineB, ops);

        const planA = engineA._debug_buildPlanFromTrace();
        const planB = engineB._debug_buildPlanFromTrace();

        expect(planA.eventKeys).toEqual(planB.eventKeys);
        expect(engineA._debug_simulateCommitPlan(planA)).toEqual(
          engineB._debug_simulateCommitPlan(planB),
        );
      }),
      { numRuns: 60 },
    );
  });

  it("preserves monotonic version counts", () => {
    fc.assert(
      fc.property(rawOpsArb, (rawOps) => {
        const ops = normalizeOps(rawOps);
        const engine = new Engine();
        applyOps(engine, ops);

        const snapshot = engine._debugSnapshot();
        const counts = countOps(ops);

        for (const [locId, count] of Object.entries(counts.locLogical)) {
          expect(snapshot.locs[locId].locLogicalVersion).toBe(count);
        }
        for (const [locId, count] of Object.entries(counts.locCommit)) {
          expect(snapshot.locs[locId].locVersion).toBe(count);
        }
        for (const [baseId, count] of Object.entries(counts.baseCommit)) {
          expect(snapshot.bases[baseId].baseCommitVersion).toBe(count);
        }
        for (const [baseId, mutIds] of Object.entries(counts.baseMutIds)) {
          expect(snapshot.bases[baseId].committedMutations).toEqual(mutIds);
        }
      }),
      { numRuns: 60 },
    );
  });

  it("orders rng draws by drawNonce for each opNonce", () => {
    const rngOnlyArb = fc
      .array(opNonceArb, { minLength: 1, maxLength: 25 })
      .map((opNonces) =>
        opNonces.map((opNonce) => ({ kind: "rngDraw", opNonce })),
      );

    fc.assert(
      fc.property(rngOnlyArb, (rawOps) => {
        const ops = normalizeOps(rawOps);
        const engine = new Engine();
        applyOps(engine, ops);

        const plan = engine._debug_buildPlanFromTrace();
        const rngEvents = plan.orderedEvents.filter(
          (event) => event.key.kind === "rng_draw",
        );

        const byOpNonce = new Map<number, number[]>();
        for (const event of rngEvents) {
          const list = byOpNonce.get(event.key.opNonce) ?? [];
          list.push(event.key.drawNonce);
          byOpNonce.set(event.key.opNonce, list);
        }

        for (const draws of byOpNonce.values()) {
          const sorted = draws.slice().sort((a, b) => a - b);
          expect(draws).toEqual(sorted);
        }
      }),
      { numRuns: 60 },
    );
  });
});

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
