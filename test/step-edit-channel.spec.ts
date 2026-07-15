/**
 * The StepEditChannel (task #98 phase 6) — the ONE step-object edit seam
 * (`docs/step-object-design.md §5`). Pins the §5.1 typed surface, the §5.2
 * refusal semantics (typed codes, never a throw; rollback is the identity), and
 * the record-request-not-mutate discipline (§5.1 — nothing mutates a live facet).
 *
 * The NULL-EDIT gate at unit altitude: a REFUSED merge and a rollback change
 * NOTHING — the pending request list returns to empty, no facet is touched. The
 * full-stack step-altitude null differential (streams byte-identical with the
 * partition facet reified + an identity edit routed) is the GPU tool
 * tools/t-step-edit-null.ts.
 *
 * Pure logic (no GPU) — runs in the cpu project.
 */
import { describe, expect, it } from "vitest";
import type { IslandId } from "../src/schedule/moves/fuse";
import { makeStepEditChannel } from "../src/schedule/moves/step-edit-channel";

// Island ids at this altitude are region UIDs (branded strings) or numeric
// indices; the channel reads them structurally. Use numeric-index neighbors so
// the conservative no-flow convexity default is exercised.
const isl = (n: number) => `${n}` as unknown as IslandId;

describe("StepEditChannel — §5.1 surface, §5.2 refusals (task #98 phase 6)", () => {
  it("exposes EXACTLY the §5.1 method surface (no requestSlotRebind in v1)", () => {
    const ch = makeStepEditChannel();
    expect(typeof ch.requestMerge).toBe("function");
    expect(typeof ch.requestSplit).toBe("function");
    expect(typeof ch.requestRecompute).toBe("function");
    expect(typeof ch.requestRingDepth).toBe("function");
    expect(typeof ch.rollback).toBe("function");
    expect(typeof ch.pauseAtBoundary).toBe("function");
    // v1 fixes slot sources at declaration — no rebind method (§5.1 / §10 Q1).
    expect(
      (ch as unknown as Record<string, unknown>).requestSlotRebind,
    ).toBeUndefined();
  });

  it("records a requested decision, does NOT mutate a live facet (record-request-not-mutate)", () => {
    const ch = makeStepEditChannel();
    expect(ch.pending).toEqual([]);
    const out = ch.requestMerge(isl(3), isl(4)); // adjacent → convex default
    expect(out.kind).toBe("accepted");
    // The request is QUEUED (a pending re-record request), not applied.
    expect(ch.pending.length).toBe(1);
    expect(ch.pending[0].kind).toBe("merge");
  });

  it("an ILLEGAL merge is a typed refusal, never a throw; leaves the declaration untouched (§5.2)", () => {
    const ch = makeStepEditChannel();
    // Non-adjacent, no island-flow → convexity cannot be witnessed → refused.
    const out = ch.requestMerge(isl(1), isl(9));
    expect(out.kind).toBe("refused");
    if (out.kind === "refused") expect(out.code).toBe("MERGE_REFUSED");
    // The refusal is the identity: nothing queued.
    expect(ch.pending).toEqual([]);
  });

  it("island-flow convexity gates the merge (§5.2, the fuse.ts mergeConvex rule)", () => {
    const flow = [{ from: isl(10), to: isl(20) }];
    const ch = makeStepEditChannel({ islandFlow: flow });
    // Edge present → convex → accepted.
    expect(ch.requestMerge(isl(10), isl(20)).kind).toBe("accepted");
    // No edge → not convex → refused.
    const ch2 = makeStepEditChannel({ islandFlow: flow });
    expect(ch2.requestMerge(isl(10), isl(30)).kind).toBe("refused");
  });

  it("rollback is the IDENTITY: a recorded merge, rolled back, returns to empty (§5.2)", () => {
    const ch = makeStepEditChannel();
    const out = ch.requestMerge(isl(5), isl(6));
    expect(out.kind).toBe("accepted");
    if (out.kind === "accepted") ch.rollback(out.handle);
    expect(ch.pending).toEqual([]);
  });

  it("rollback works uniformly across facets (split/recompute/ring, not just merge)", () => {
    const ch = makeStepEditChannel();
    const sp = ch.requestSplit("region:fused(1+2)", 2);
    const rc = ch.requestRecompute(0xabc, "retain");
    const rg = ch.requestRingDepth(3);
    expect(ch.pending.length).toBe(3);
    if (rc.kind === "accepted") ch.rollback(rc.handle); // roll the middle one
    expect(ch.pending.map((r) => r.kind)).toEqual(["split", "ringDepth"]);
    if (sp.kind === "accepted") ch.rollback(sp.handle);
    if (rg.kind === "accepted") ch.rollback(rg.handle);
    expect(ch.pending).toEqual([]);
    // An unknown handle is a no-op (does not throw, does not drop anything).
    ch.requestRingDepth(2);
    ch.rollback("bogus-handle");
    expect(ch.pending.length).toBe(1);
  });

  it("requestSplit: a member-boundary cut is legal; a negative cut is refused", () => {
    const ch = makeStepEditChannel();
    expect(ch.requestSplit("region:fused(1+2)", 3).kind).toBe("accepted");
    const bad = ch.requestSplit("region:fused(1+2)", -1);
    expect(bad.kind).toBe("refused");
    if (bad.kind === "refused") expect(bad.code).toBe("SPLIT_REFUSED");
  });

  it("requestRecompute: a valid segment fp toggles; the 0 sentinel is refused (§5.2)", () => {
    const ch = makeStepEditChannel();
    expect(ch.requestRecompute(0x16e9a591, "retain").kind).toBe("accepted");
    const bad = ch.requestRecompute(0, "recompute");
    expect(bad.kind).toBe("refused");
    if (bad.kind === "refused") expect(bad.code).toBe("RECOMPUTE_ILLEGAL");
  });

  it("requestRingDepth: K≥1 within budget accepted; K<1 or over-cap refused (§5.2 / risk 4)", () => {
    const ch = makeStepEditChannel();
    expect(ch.requestRingDepth(2).kind).toBe("accepted");
    const zero = ch.requestRingDepth(0);
    expect(zero.kind).toBe("refused");
    if (zero.kind === "refused") expect(zero.code).toBe("RING_OUT_OF_BUDGET");
    const huge = ch.requestRingDepth(999);
    expect(huge.kind).toBe("refused");
    if (huge.kind === "refused") expect(huge.code).toBe("RING_OUT_OF_BUDGET");
  });

  it("pauseAtBoundary is RESERVED: a typed NOT_IMPLEMENTED refusal, zero behavioral surface (§5.1)", () => {
    const ch = makeStepEditChannel();
    const h = ch.pauseAtBoundary(0x16e9a591);
    expect(h.kind).toBe("not-implemented");
    // Reserving it did NOT queue anything — zero behavioral surface.
    expect(ch.pending).toEqual([]);
  });

  it("the S3 agreement seam: a legal merge with applyMerge drives the detector's own merge (the wiring boundary)", () => {
    const applied: Array<{ a: string; b: string; region: string }> = [];
    const ch = makeStepEditChannel({
      islandFlow: [{ from: isl(1), to: isl(2) }],
      applyMerge: (a, b, region) =>
        applied.push({
          a: a as unknown as string,
          b: b as unknown as string,
          region,
        }),
    });
    const out = ch.requestMerge(isl(1), isl(2));
    expect(out.kind).toBe("accepted");
    // The channel drove the (stubbed) detector merge exactly once — the seam a
    // live fuseGesture binding fills. STOP before live partition execution.
    expect(applied.length).toBe(1);
    expect(applied[0].a).toBe("1");
    expect(applied[0].b).toBe("2");
  });

  it("applyMerge that THROWS rolls back atomically (§5.2 — a refused edit is the identity)", () => {
    const ch = makeStepEditChannel({
      islandFlow: [{ from: isl(1), to: isl(2) }],
      applyMerge: () => {
        throw new Error("realizer refused");
      },
    });
    const out = ch.requestMerge(isl(1), isl(2));
    expect(out.kind).toBe("refused");
    if (out.kind === "refused") expect(out.code).toBe("MERGE_REFUSED");
    // Atomic rollback: nothing left queued.
    expect(ch.pending).toEqual([]);
  });
});
