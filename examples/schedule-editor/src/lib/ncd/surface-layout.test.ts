import { describe, expect, it } from "vitest";
import attention from "../../../public/data/ncd/attention-naive.term.json";
import matmul from "../../../public/data/ncd/tiled-matmul.term.json";
import { napkinCost } from "./model";
import {
  surfaceColumnCosts,
  surfaceColumnX,
  surfaceLanes,
} from "./surface-layout";
import type { NcdTerm } from "./types";

const TERMS = [attention as NcdTerm, matmul as NcdTerm];

describe("term-derived NCD surface", () => {
  it.each(TERMS)(
    "aligns cumulative column H with the napkin cost for $name",
    (term) => {
      const columns = surfaceColumnCosts(term);
      expect(columns.at(-1)?.cumulative.l1).toBe(
        napkinCost(term).transferByLevel.l1,
      );
    },
  );

  it("derives all placement without document coordinates", () => {
    const term = attention as NcdTerm;
    expect(surfaceColumnX(term, 0)).toBeLessThan(surfaceColumnX(term, 1));
    expect(surfaceLanes(term).map((lane) => lane.y)).toEqual(
      [...surfaceLanes(term).map((lane) => lane.y)].sort((a, b) => a - b),
    );

    const documentKeys = new Set<string>();
    JSON.stringify(term, (key, value) => {
      documentKeys.add(key);
      return value;
    });
    expect(documentKeys.has("x")).toBe(false);
    expect(documentKeys.has("y")).toBe(false);
  });
});
