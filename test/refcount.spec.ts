import { beforeEach, describe, expect, it } from "vitest";
import { rcGet, rcRelease, rcReset, rcRetain } from "../src/graph/refcount";

describe("refcount", () => {
  beforeEach(() => {
    rcReset();
  });

  it("rcRetain / rcRelease / rcGet track counts", () => {
    expect(rcGet(1)).toBe(-1); // unknown
    rcRetain(1, "test.create");
    expect(rcGet(1)).toBe(1);
    rcRetain(1, "test.tensor");
    expect(rcGet(1)).toBe(2);
    rcRelease(1, "test.dispose");
    expect(rcGet(1)).toBe(1);
    rcRelease(1, "test.nodeClear");
    expect(rcGet(1)).toBe(0);
  });

  it("rcRelease returns new count", () => {
    rcRetain(1, "test");
    rcRetain(1, "test");
    expect(rcRelease(1, "test")).toBe(1);
    expect(rcRelease(1, "test")).toBe(0);
  });

  it("rcRelease on unknown storage returns -1", () => {
    expect(rcRelease(99999, "test.unknown")).toBe(-1);
  });

  it("rcRelease can go negative (double-release indicator)", () => {
    rcRetain(1, "test");
    rcRelease(1, "test");
    expect(rcGet(1)).toBe(0);
    rcRelease(1, "test.double");
    expect(rcGet(1)).toBe(-1);
  });

  it("rcReset clears all counts", () => {
    rcRetain(1, "test");
    rcRetain(2, "test");
    rcReset();
    expect(rcGet(1)).toBe(-1);
    expect(rcGet(2)).toBe(-1);
  });
});
