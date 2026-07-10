/**
 * Regenerate the fusion-decision fixture (test/fixtures/fusion-decisions.json)
 * from the CURRENT detector. Run ONLY when a decision change is INTENTIONAL
 * (e.g. the I2b gap-spanning extension); the diff of the fixture is the
 * reviewable record of exactly which decisions changed. The null-form policy
 * re-expression (I2a) must NOT need a regeneration.
 *
 * Run: npx tsx tools/gen-fusion-corpus-fixture.ts
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildCorpus,
  decisionsFor,
} from "../test/helpers/fusion-corpus";

const here = path.dirname(fileURLToPath(import.meta.url));
const out = buildCorpus().map(decisionsFor);
const file = path.join(here, "..", "test", "fixtures", "fusion-decisions.json");
fs.mkdirSync(path.dirname(file), { recursive: true });
fs.writeFileSync(file, `${JSON.stringify(out, null, 2)}\n`);
console.log(`wrote ${file} (${out.length} cases)`);
process.exit(0);
