/**
 * Semantic Derivation stratum (Crystal Campaign 3, Phase 0 — the walking
 * skeleton). An op's meaning is a first-class DATA term; its CPU reference body,
 * its gradient, and its WGSL all DERIVE from that one source. See
 * docs/semantic-derivation-design.md.
 */

export * from "./adjoint";
export * from "./catalog";
export * from "./composite";
export * from "./emit-rt";
export * from "./erf";
export * from "./expr";
export * from "./index-map";
export * from "./interpret";
export * from "./reduction";
