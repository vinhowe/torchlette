/**
 * Environment access that is safe in BOTH Node and the browser.
 *
 * `process` does not exist in browsers; a bare `process.env.X` read crashes
 * the bundle at the first call (the entire browser test suite failed with
 * "process is not defined" after env-gated debug/feature checks accumulated
 * in hot paths). Every environment read in src/ goes through this module —
 * in the browser all flags simply read as undefined (defaults apply).
 *
 * Note for tree-shaking/bundlers: this is a runtime check, not a build-time
 * define; flags stay dynamic in Node (tests set them per-subprocess).
 */
export const ENV: Record<string, string | undefined> =
  typeof process !== "undefined" && process.env
    ? (process.env as Record<string, string | undefined>)
    : {};
