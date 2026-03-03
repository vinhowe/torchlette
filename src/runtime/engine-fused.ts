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
  FusedRMSNormConfig,
} from "../backend/types";
import { createPendingRef, type LazyRef } from "../engine/lazy-types";
import { createLazyIRNode } from "../engine/node-factory";

interface FusedOpResult {
  ref: LazyRef;
  shape: number[];
}

/** Create a fused op node and return its pending ref + output shape. */
function makeFusedOp(
  op: string,
  inputs: LazyRef[],
  shape: number[],
  device: DeviceKind,
  config?: unknown,
): FusedOpResult {
  const node = createLazyIRNode(op, inputs, shape, "f32", device, config);
  return { ref: createPendingRef(node), shape };
}

// Cross-entropy
export function fusedCrossEntropyForwardOp(
  logitsRef: LazyRef,
  targetsRef: LazyRef,
  device: DeviceKind,
  config: FusedCrossEntropyConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedCrossEntropyForward",
    [logitsRef, targetsRef],
    [config.batchSize],
    device,
    config,
  );
}

export function fusedCrossEntropyBackwardOp(
  logitsRef: LazyRef,
  targetsRef: LazyRef,
  gradOutputRef: LazyRef,
  device: DeviceKind,
  config: FusedCrossEntropyConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedCrossEntropyBackward",
    [logitsRef, targetsRef, gradOutputRef],
    [config.batchSize, config.vocabSize],
    device,
    config,
  );
}

// LayerNorm
export function fusedLayerNormForwardOp(
  xRef: LazyRef,
  weightRef: LazyRef,
  biasRef: LazyRef,
  xShape: number[],
  device: DeviceKind,
  config: FusedLayerNormConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedLayerNormForward",
    [xRef, weightRef, biasRef],
    xShape.slice(),
    device,
    config,
  );
}

export function fusedLayerNormBackwardGradXOp(
  gradOutputRef: LazyRef,
  xRef: LazyRef,
  weightRef: LazyRef,
  xShape: number[],
  device: DeviceKind,
  config: FusedLayerNormConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedLayerNormBackwardGradX",
    [gradOutputRef, xRef, weightRef],
    xShape.slice(),
    device,
    config,
  );
}

export function fusedLayerNormBackwardGradWeightBiasOp(
  gradOutputRef: LazyRef,
  xRef: LazyRef,
  device: DeviceKind,
  config: FusedLayerNormConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedLayerNormBackwardGradWeightBias",
    [gradOutputRef, xRef],
    [config.featureDim],
    device,
    config,
  );
}

// Attention
export function fusedAttentionForwardOp(
  qRef: LazyRef,
  kRef: LazyRef,
  vRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedAttentionForward",
    [qRef, kRef, vRef],
    [config.batchSize, config.numHeads, config.seqLen, config.headDim],
    device,
    config,
  );
}

export function extractAttentionLogsumexpOp(
  fwdOutputRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  return makeFusedOp(
    "extractAttentionLogsumexp",
    [fwdOutputRef],
    [config.batchSize, config.numHeads, config.seqLen],
    device,
  );
}

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
  return makeFusedOp(
    "fusedAttentionBackward",
    [qRef, kRef, vRef, logsumexpRef, dORef, outputRef],
    [config.batchSize, config.numHeads, config.seqLen, config.headDim],
    device,
    config,
  );
}

export function extractAttentionDKOp(
  bwdDQRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  return makeFusedOp(
    "extractAttentionDK",
    [bwdDQRef],
    [config.batchSize, config.numHeads, config.seqLen, config.headDim],
    device,
  );
}

export function extractAttentionDVOp(
  bwdDQRef: LazyRef,
  device: DeviceKind,
  config: FusedAttentionConfig,
): FusedOpResult {
  return makeFusedOp(
    "extractAttentionDV",
    [bwdDQRef],
    [config.batchSize, config.numHeads, config.seqLen, config.headDim],
    device,
  );
}

// RMSNorm
export function fusedRMSNormForwardOp(
  xRef: LazyRef,
  weightRef: LazyRef,
  xShape: number[],
  device: DeviceKind,
  config: FusedRMSNormConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedRMSNormForward",
    [xRef, weightRef],
    xShape.slice(),
    device,
    config,
  );
}

export function fusedRMSNormBackwardGradXOp(
  gradOutputRef: LazyRef,
  xRef: LazyRef,
  weightRef: LazyRef,
  xShape: number[],
  device: DeviceKind,
  config: FusedRMSNormConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedRMSNormBackwardGradX",
    [gradOutputRef, xRef, weightRef],
    xShape.slice(),
    device,
    config,
  );
}

export function fusedRMSNormBackwardGradWeightOp(
  gradOutputRef: LazyRef,
  xRef: LazyRef,
  weightRef: LazyRef,
  device: DeviceKind,
  config: FusedRMSNormConfig,
): FusedOpResult {
  return makeFusedOp(
    "fusedRMSNormBackwardGradWeight",
    [gradOutputRef, xRef, weightRef],
    [config.featureDim],
    device,
    config,
  );
}

// LayerNorm side output extraction
export function extractLnBwdGradBiasOp(
  gradWeightRef: LazyRef,
  device: DeviceKind,
  featureDim: number,
): FusedOpResult {
  return makeFusedOp(
    "extractLnBwdGradBias",
    [gradWeightRef],
    [featureDim],
    device,
    { featureDim },
  );
}
