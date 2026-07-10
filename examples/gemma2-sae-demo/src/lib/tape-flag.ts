/**
 * Enables step-tape replay in the browser. MUST be the FIRST import of the
 * worker entry: torchlette reads TORCHLETTE_STEP_TAPE into module-load
 * consts, and this dependency-free module is guaranteed by ES import order
 * to evaluate before torchlette's module graph does (see src/core/env.ts).
 */
(globalThis as { __TORCHLETTE_ENV__?: Record<string, string> }).__TORCHLETTE_ENV__ = {
  TORCHLETTE_STEP_TAPE: "1",
};
