/**
 * Fusion-decision differential (islands I2 null gate).
 *
 * Pins the detector's decisions over a corpus covering its whole branch
 * space (consecutive runs, independent/dependent gaps, components,
 * multi-output promotion, buffer-limit splits, singleton batching,
 * exclusions, scalars, readiness). The committed fixture is the byte-stable
 * record:
 *  - the I2a policy re-expression must reproduce it EXACTLY (identical
 *    decisions → the re-expression is proven safe);
 *  - the I2b gap-spanning extension regenerates it INTENTIONALLY
 *    (`npx tsx tools/gen-fusion-corpus-fixture.ts`) with the decision delta
 *    reviewable in the fixture diff.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildCorpus, decisionsFor } from "./helpers/fusion-corpus";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixturePath = path.join(here, "fixtures", "fusion-decisions.json");

describe("fusion decision corpus (islands null gate)", () => {
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8")) as Array<{
    name: string;
  }>;
  const cases = buildCorpus();

  it("fixture covers the corpus exactly", () => {
    expect(fixture.map((f) => f.name)).toEqual(cases.map((c) => c.name));
  });

  for (let i = 0; i < cases.length; i++) {
    it(`decisions byte-stable: ${cases[i].name}`, () => {
      expect(decisionsFor(cases[i])).toEqual(fixture[i]);
    });
  }
});
