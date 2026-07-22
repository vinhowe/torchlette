/**
 * DEMOTED (P4b-R Phase R1, 2026-07-22): the step-tape is NO LONGER enabled.
 *
 * The demos' default decode path is the unrolled-K BLOCK (`decodeBlock`), which
 * calls straight into the executor's build-from-IR + observed-liveness harvest
 * and NEVER routes through the tape-replay `api.capture` loop. The census
 * (`tools/t-p4b-r-census.ts`) measured the block byte-identical with identical
 * submit counts and zero uncaptured GPU errors at STEP_TAPE=0 vs =1 — the tape
 * added ZERO functional and ZERO performance value; its only effect was
 * populating a self-referential cross-plan-edge UAF guard (`producers` 0→1).
 * Setting the flag off makes the correct-and-slow tape-free path the only one.
 *
 * The file is retained for the ES import-order contract only (torchlette reads
 * env into module-load consts; this dependency-free module still evaluates
 * first). It sets NOTHING now; it is removed entirely when the STEP_TAPE flag
 * dies in Phase R3.
 */
(globalThis as { __TORCHLETTE_ENV__?: Record<string, string> }).__TORCHLETTE_ENV__ = {};
