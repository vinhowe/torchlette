import type { Tensor as RuntimeTensor } from "./runtime/tensor";
import { TidyDispatchMode } from "./runtime/engine";
import { storageTracker } from "./engine/lazy";
import { shapesEqual } from "./core/shape";
import type { Torchlette } from "./frontend";
import type { Tensor } from "./frontend-tensor";
import type { AutogradNode, GetSavedFn } from "./frontend-types";

export async function backwardImpl(
  torch: Torchlette,
  a: Tensor,
  grad?: Tensor,
): Promise<void> {
  torch._assertUsable(a);
  if (grad) {
    torch._assertUsable(grad);
    if (!shapesEqual(grad.shape, a.shape)) {
      throw new Error("backward grad shape mismatch");
    }
  }

  return torch._runEntryPoint(async () => {
    // Wrap the entire backward pass in a TidyDispatchMode to auto-dispose
    // unwrapped RuntimeTensors (gradient accumulation intermediates,
    // non-retained grads) at scope exit. Leaf/retained grads survive via
    // wrap() → markEscaped().
    const tidyMode = new TidyDispatchMode();
    torch.runtime.pushDispatchMode(tidyMode);
    try {
      return await torch.engine.runWithAsyncScope(async () => {
      const seed = grad ? grad._unwrap() : torch._seedGrad(a);

      // Use Tensor as key for gradient accumulation
      const gradMap = new Map<Tensor, RuntimeTensor>();
      gradMap.set(a, seed);

      const ordered: AutogradNode[] = [];
      const visited = new Set<AutogradNode>();

      // Graph traversal uses inputs array
      const visit = (node: AutogradNode) => {
        if (visited.has(node)) return;
        for (const input of node.inputs) {
          const inputNode = input._gradNode();
          if (inputNode) {
            visit(inputNode);
          }
        }
        visited.add(node);
        ordered.push(node);
      };

      const rootNode = a._gradNode();
      if (rootNode) visit(rootNode);

      // Force all tensors needed for backward
      // Skip disposed/materialized tensors to avoid redundant plan building
      const tensorsToForce: RuntimeTensor[] = [];
      if (!seed.isMaterialized()) tensorsToForce.push(seed);
      for (const node of ordered) {
        for (const input of node.inputs) {
          const rt = input._unwrap();
          if (!rt.disposed && !rt.isMaterialized()) {
            tensorsToForce.push(rt);
          }
        }
      }
      if (tensorsToForce.length > 0) {
        await torch.runtime.forceAll(...tensorsToForce);
      }

      // ========================================================================
      // UNIFIED BACKWARD EXECUTION
      // ========================================================================
      // Phase A: Collect ALL saved tensors from ALL nodes (triggers lazy
      //          recompute graph construction for checkpointed tensors)
      // Phase B: Force ALL in ONE merged plan - this ensures checkpoint
      //          boundaries appear together, enabling proper segmentation
      // Phase C: Run backward functions with materialized tensors
      // ========================================================================

      const allUnpackedTensors = new Map<AutogradNode, Tensor[]>();

      // Phase A: Collect all unpacked tensors
      for (let i = ordered.length - 1; i >= 0; i -= 1) {
        const node = ordered[i];
        for (const slot of node.savedSlots) {
          torch.engine._debug_useSavedTensor(slot.record);
        }
        const unpackedTensors: Tensor[] = [];
        for (const slot of node.savedSlots) {
          const tensor = slot.unpackHook(slot.packed);
          unpackedTensors.push(tensor);
        }
        allUnpackedTensors.set(node, unpackedTensors);
      }

      // Phase B: Force all saved tensors in ONE merged plan
      const allPending: RuntimeTensor[] = [];
      for (const tensors of allUnpackedTensors.values()) {
        for (const t of tensors) {
          allPending.push(t._unwrap());
        }
      }
      if (allPending.length > 0) {
        await torch.runtime.forceAll(...allPending);
      }

      // Phase C: Run ALL backward functions lazily (no forcing), then force
      // all final gradients in a single forceAll() at the end.
      // Gradient accumulation intermediates (old existing + old gradIn after
      // runtime.add) are tracked by TidyDispatchMode and disposed at scope exit.
      for (let i = ordered.length - 1; i >= 0; i -= 1) {
        const node = ordered[i];
        const gradOutTensor = gradMap.get(node.output);
        if (!gradOutTensor) continue;

        const unpackedTensors = allUnpackedTensors.get(node) || [];
        const getSaved: GetSavedFn = (idx: number): Tensor => {
          if (idx >= unpackedTensors.length) {
            throw new Error(`No saved tensor at index ${idx}`);
          }
          return unpackedTensors[idx];
        };

        // Fire backward dispatch hooks
        for (const hook of torch._backwardDispatchHooks) {
          hook({ output: node.output, inputs: node.inputs, label: node.label });
        }

        torch.runtime.startIntermediateTracking();
        let gradsIn: Array<RuntimeTensor | null>;
        try {
          gradsIn = node.backward(gradOutTensor, getSaved);
        } catch (_e: any) {
          torch.runtime.stopIntermediateTracking();
          await torch.runtime.force(gradOutTensor);
          torch.runtime.startIntermediateTracking();
          gradsIn = node.backward(gradOutTensor, getSaved);
        }

        const trackedIntermediates = torch.runtime.stopIntermediateTracking();

        const keepSet = new Set(gradsIn.filter((g): g is RuntimeTensor => g !== null));
        for (const tensor of trackedIntermediates) {
          if (!keepSet.has(tensor) && !tensor.disposed) {
            tensor.dispose();
          }
        }

        // Track which gradIn RuntimeTensors are actually used (for requiresGrad inputs).
        // We need this because sumToShape may return the SAME RuntimeTensor for multiple
        // inputs when shapes match, so we can't dispose a non-requiresGrad gradIn if
        // it's also used for a requiresGrad input.
        const usedGradIns = new Set<RuntimeTensor>();
        const unusedGradIns: RuntimeTensor[] = [];
        for (let idx = 0; idx < node.inputs.length; idx += 1) {
          const input = node.inputs[idx];
          const gradIn = gradsIn[idx];
          if (!gradIn) continue;
          if (!input.requiresGrad) {
            unusedGradIns.push(gradIn);
            continue;
          }

          usedGradIns.add(gradIn);
          const existing = gradMap.get(input);
          if (existing) {
            const accumulated = torch.runtime.add(existing, gradIn);
            gradMap.set(input, accumulated);
            // old existing and gradIn are tracked by TidyDispatchMode (not wrapped/escaped)
            // and will be disposed at backward scope exit.
          } else {
            gradMap.set(input, gradIn);
          }
        }
        // Dispose gradients for non-requiresGrad inputs that aren't shared
        // with any requiresGrad input. Without this, the RuntimeTensor stays
        // in pendingTensorsByNodeId, gets materialized at markStep, and is
        // never marked unreachable — leaking one StorageHandle per step.
        for (const g of unusedGradIns) {
          if (!g.disposed && !usedGradIns.has(g)) {
            g.dispose();
          }
        }
      }

      // Dispose checkpoint-recomputed tensors (no longer needed after Phase C).
      // Build set of all node inputs (parameters + user tensors) to protect them.
      // During checkpoint recomputation, parameters flow through fn(...inputs)
      // as the SAME Tensor objects and get recaptured — we must not dispose these.
      const protectedTensors = new Set<Tensor>();
      for (const node of ordered) {
        for (const input of node.inputs) {
          protectedTensors.add(input);
        }
      }
      for (const [node, tensors] of allUnpackedTensors.entries()) {
        for (let idx = 0; idx < tensors.length; idx++) {
          const unpacked = tensors[idx];
          const packed = node.savedSlots[idx]?.packed;
          // Non-checkpoint: packed === unpacked (identity hook) -> skip
          // Checkpoint: packed is a placeholder, unpacked is a Tensor
          //   - If unpacked is a parameter/input tensor -> skip (protected)
          //   - If unpacked is a recomputed intermediate -> dispose
          if (unpacked === packed) {
            // identity
          } else if (unpacked.isDisposed) {
            // already disposed (e.g. by tidy in checkpoint recomputation)
          } else if (protectedTensors.has(unpacked)) {
            // protected
          } else {
            unpacked.dispose();
          }
        }
      }
      allUnpackedTensors.clear();

      // Force ALL final gradients in one merged plan.
      const allGrads = [...gradMap.values()].filter(
        (g) => !g.isMaterialized() && !g.disposed
      );
      if (allGrads.length > 0) {
        await torch.runtime.forceAll(...allGrads);
      }
      storageTracker.destroyUnreachable();

      // Store final gradients on leaf tensors and retained non-leaf tensors
      // Mark these tensors as "kept" so they survive async scope cleanup
      for (const [tensor, gradTensor] of gradMap) {
        const isLeaf = tensor.requiresGrad && !tensor._gradNode();
        const shouldRetain = tensor.isRetainGrad;

        if (isLeaf || shouldRetain) {
          // Wrap and keep the gradient tensor before async scope exits
          const gradWrapper = torch._wrap(gradTensor, false);
          torch.keep(gradWrapper);
          tensor._setGrad(gradWrapper);
        }
        // Non-retained grads: not wrapped, so TidyDispatchMode disposes them at scope exit
      }

      // Clean up saved tensors and autograd graph after backward
      // This is critical for memory management - without it, saved tensors
      // accumulate across training steps causing out-of-memory errors.

      // Build set of autograd node outputs - these are forward pass intermediates
      // that are safe to dispose.
      const forwardIntermediates = new Set<Tensor>();
      for (const node of ordered) {
        forwardIntermediates.add(node.output);
      }

      // Build set of tensors that must NOT be disposed:
      // - Parameters (leaf tensors with requiresGrad)
      // - User-held inputs that are not forward intermediates (e.g., x, target)
      const preserved = new Set<Tensor>();
      for (const [tensor, _grad] of gradMap) {
        const isLeaf = tensor.requiresGrad && !tensor._gradNode();
        if (isLeaf) preserved.add(tensor);
      }
      for (const node of ordered) {
        for (const input of node.inputs) {
          if (!forwardIntermediates.has(input) && !torch._compileCreatedTensors.has(input)) {
            preserved.add(input);
          }
        }
      }

      // Collect forward intermediates to dispose (excluding preserved tensors)
      const toDispose = new Set<Tensor>();
      for (const node of ordered) {
        if (!preserved.has(node.output)) {
          toDispose.add(node.output);
        }
      }

      for (const node of ordered) {
        // Dispose saved tensors that are internal intermediates (e.g., autocast
        // casts). Only dispose if the saved tensor is NOT a preserved tensor
        // (parameter or user-held input) and is NOT already in toDispose.
        for (const slot of node.savedSlots) {
          const savedTensor = slot.packed as Tensor;
          if (savedTensor && typeof savedTensor.dispose === "function") {
            if (!preserved.has(savedTensor) && !savedTensor.disposed) {
              toDispose.add(savedTensor);
            }
          }
        }
        node.savedSlots.length = 0;
        node.output._setGradNode(null);
        node.inputs.length = 0;
      }

      // Dispose all collected tensors
      for (const tensor of toDispose) {
        if (!tensor.disposed) {
          tensor.dispose();
        }
      }

    });
    } finally {
      torch.runtime.popDispatchMode();
      tidyMode.disposeNonEscaped();
    }
  });
}
