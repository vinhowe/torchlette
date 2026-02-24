/**
 * Fused kernel op helpers extracted from RuntimeEngine.
 *
 * Each function creates a LazyIRNode for a fused kernel op and returns
 * the node, pending ref, and output shape. The RuntimeEngine method
 * is a thin wrapper that passes the result to createAndTrack().
 */

import type {
  DeviceKind,
  FusedAttentionConfig,
  FusedCrossEntropyConfig,
  FusedLayerNormConfig,
} from "../backend/types";
import {
  createLazyIRNode,
  createPendingRef,
  type LazyRef,
} from "../engine/lazy";

interface FusedOpResult {
  ref: LazyRef;
  shape: number[];
}

/**
 * Fused cross-entropy forward: logits [B,V] + targets [B] → per-sample loss [B].
 */
export function fusedCrossEntropyForwardOp(
  logitsRef: LazyRef,
  targetsRef: LazyRef,
  device: DeviceKind,
  config: FusedCrossEntropyConfig,
): FusedOpResult {
  const shape = [config.batchSize];
  const node = createLazyIRNode(
    "fusedCrossEntropyForward",
    [logitsRef, targetsRef],
    shape,
    "f32",
    device,
    config,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Fused cross-entropy backward: logits [B,V] + targets [B] + grad [B] → grad_logits [B,V].
 */
export function fusedCrossEntropyBackwardOp(
  logitsRef: LazyRef,
  targetsRef: LazyRef,
  gradOutputRef: LazyRef,
  device: DeviceKind,
  config: FusedCrossEntropyConfig,
): FusedOpResult {
  const shape = [config.batchSize, config.vocabSize];
  const node = createLazyIRNode(
    "fusedCrossEntropyBackward",
    [logitsRef, targetsRef, gradOutputRef],
    shape,
    "f32",
    device,
    config,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Fused LayerNorm forward: x [N,D] + weight [D] + bias [D] → output [N,D].
 */
export function fusedLayerNormForwardOp(
  xRef: LazyRef,
  weightRef: LazyRef,
  biasRef: LazyRef,
  xShape: number[],
  device: DeviceKind,
  config: FusedLayerNormConfig,
): FusedOpResult {
  const shape = xShape.slice();
  const node = createLazyIRNode(
    "fusedLayerNormForward",
    [xRef, weightRef, biasRef],
    shape,
    "f32",
    device,
    config,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Fused LayerNorm backward gradX: grad [N,D] + x [N,D] + weight [D] → gradX [N,D].
 */
export function fusedLayerNormBackwardGradXOp(
  gradOutputRef: LazyRef,
  xRef: LazyRef,
  weightRef: LazyRef,
  xShape: number[],
  device: DeviceKind,
  config: FusedLayerNormConfig,
): FusedOpResult {
  const shape = xShape.slice();
  const node = createLazyIRNode(
    "fusedLayerNormBackwardGradX",
    [gradOutputRef, xRef, weightRef],
    shape,
    "f32",
    device,
    config,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Fused LayerNorm backward gradWeight+gradBias.
 * Returns gradWeight as main output. gradBias extracted via extractLnBwdGradBiasOp.
 */
export function fusedLayerNormBackwardGradWeightBiasOp(
  gradOutputRef: LazyRef,
  xRef: LazyRef,
  device: DeviceKind,
  config: FusedLayerNormConfig,
): FusedOpResult {
  const shape = [config.featureDim];
  const node = createLazyIRNode(
    "fusedLayerNormBackwardGradWeightBias",
    [gradOutputRef, xRef],
    shape,
    "f32",
    device,
    config,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Fused attention forward: Q,K,V [B,H,N,D] → O [B,H,N,D].
 * Logsumexp is extracted via extractAttentionLogsumexpOp.
 */
export function fusedAttentionForwardOp(
  qRef: LazyRef,
  kRef: LazyRef,
  vRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
  const node = createLazyIRNode(
    "fusedAttentionForward",
    [qRef, kRef, vRef],
    shape,
    "f32",
    device,
    config,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Extract logsumexp side output [B,H,N] from fusedAttentionForward.
 */
export function extractAttentionLogsumexpOp(
  fwdOutputRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  const shape = [config.batchSize, config.numHeads, config.seqLen];
  const node = createLazyIRNode(
    "extractAttentionLogsumexp",
    [fwdOutputRef],
    shape,
    "f32",
    device,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Fused attention backward: Q,K,V,L,dO,O → dQ [B,H,N,D].
 * dK and dV are extracted via extractAttentionDKOp/DVOp.
 */
export function fusedAttentionBackwardOp(
  qRef: LazyRef,
  kRef: LazyRef,
  vRef: LazyRef,
  logsumexpRef: LazyRef,
  dORef: LazyRef,
  outputRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
  const node = createLazyIRNode(
    "fusedAttentionBackward",
    [qRef, kRef, vRef, logsumexpRef, dORef, outputRef],
    shape,
    "f32",
    device,
    config,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Extract dK side output from fusedAttentionBackward.
 */
export function extractAttentionDKOp(
  bwdDQRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
  const node = createLazyIRNode(
    "extractAttentionDK",
    [bwdDQRef],
    shape,
    "f32",
    device,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Extract dV side output from fusedAttentionBackward.
 */
export function extractAttentionDVOp(
  bwdDQRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
  const node = createLazyIRNode(
    "extractAttentionDV",
    [bwdDQRef],
    shape,
    "f32",
    device,
  );
  return { ref: createPendingRef(node), shape };
}

/**
 * Extract the gradBias side output from a fusedLayerNormBackwardGradWeightBias node.
 */
export function extractLnBwdGradBiasOp(
  gradWeightRef: LazyRef,
  device: DeviceKind,
  featureDim: number,
): FusedOpResult {
  const shape = [featureDim];
  const node = createLazyIRNode(
    "extractLnBwdGradBias",
    [gradWeightRef],
    shape,
    "f32",
    device,
    { featureDim },
  );
  return { ref: createPendingRef(node), shape };
}
