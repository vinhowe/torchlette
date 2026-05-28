/**
 * Trainer interface — the contract between the barrier protocol and the
 * underlying training implementation.
 *
 * Production: a WebGPU GPT-2 trainer that wraps the real model + Adam +
 * Nesterov outer optimizer (similar to what the current monolithic agent
 * does).
 *
 * Test: a CPU stub that does deterministic synthetic inner steps, so
 * smoke tests can exercise the full protocol end-to-end at >100 peers in
 * a single Node process without touching a GPU.
 *
 * Anchor semantics live HERE, not in the state machine. The trainer owns
 * the model params and the anchor params; the state machine only knows
 * which `AnchorRound` identifier they correspond to.
 */

/** Shape descriptor per param tensor. */
export type ParamShapes = readonly (readonly number[])[];

export interface Trainer {
  /** Shapes of each param tensor. Stable across the lifetime of the trainer. */
  paramShapes(): ParamShapes;

  /**
   * Snapshot current params as the anchor. Called once at startup (so the
   * initial random init becomes the round-0 anchor) and after every
   * successful outer step (the post-step params are the next anchor).
   *
   * Async because real implementations may need to read params from GPU.
   */
  setAnchor(): Promise<void>;

  /**
   * Run K inner training steps. `round` is informational (for logging).
   * Returns the average inner loss over the K steps (informational, the
   * state machine just attaches it to the round report).
   */
  innerSteps(round: number): Promise<number>;

  /**
   * Compute (current_params - anchor_params) per tensor. Caller MUST NOT
   * mutate the returned arrays — they may share storage with the trainer's
   * internal buffers depending on implementation.
   */
  pseudograd(): Promise<Float32Array[]>;

  /**
   * Apply outer optimization step using the averaged pseudograd from all
   * participating peers. Mutates current params AND moves the anchor to
   * the new params atomically.
   */
  applyOuterStep(avgGrad: Float32Array[]): Promise<void>;

  /**
   * Reset current params back to the anchor. Called when the quorum was
   * not met for this round so the local inner-step drift would be
   * incoherent with no peer ever seeing it.
   */
  revertToAnchor(): Promise<void>;

  /** Anchor params for outbound F16W (copies — receiver may mutate). */
  snapshotAnchor(): Promise<Float32Array[]>;

  /**
   * Apply an inbound F16W payload: overwrite params AND anchor with the
   * supplied tensors. Caller must follow with resetOptimState() to clear
   * the now-meaningless optimizer trajectory.
   */
  applyF16W(params: Float32Array[]): Promise<void>;

  /** Zero both inner and outer optimizer state (moments, velocity). */
  resetOptimState(): Promise<void>;
}
