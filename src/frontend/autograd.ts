import { shapesEqual } from "../core/shape";
import { storageTracker } from "../graph/storage-tracker";
import {
  _pendingCheckpointScopes,
  disposeCheckpointIntermediates,
} from "../nn/checkpoint";
import { TidyDispatchMode } from "../runtime/engine";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import type { Tensor } from "./tensor";
import type { Torchlette } from "./torchlette";
import type { AutogradNode, GetSavedFn } from "./types";

/**
 * Collect saved tensors from all autograd nodes. Triggers lazy recompute
 * graph construction for checkpointed segments. Returns the unpacked
 * tensors and whether any checkpoint recomputation occurred.
 */
function collectSavedTensors(
  torch: Torchlette,
  ordered: AutogradNode[],
): { unpacked: Map<AutogradNode, Tensor[]>; hasCheckpoints: boolean } {
  const unpacked = new Map<AutogradNode, Tensor[]>();

  // Suppress markEscaped during checkpoint recomputation so recomputed
  // RuntimeTensors stay tracked (not escaped) in TidyDispatchMode.
  // This ensures disposeNonEscaped() cleans them up after backward.
  //
  // Also enable checkpoint recompute tensor tracking: _wrap records every
  // FrontendTensor created during _inCheckpointRecompute into a single
  // disposal scope. This scope covers BOTH the recomputation itself AND
  // the subsequent forceAllMerged (which materializes recomputed tensors
  // and may create additional RuntimeTensors).
  const recomputeScope = new Set<Tensor>();
  torch._checkpointRecomputeTensors = recomputeScope;
  torch._inCheckpointRecompute = true;
  try {
    for (let i = ordered.length - 1; i >= 0; i -= 1) {
      const node = ordered[i];
      for (const slot of node.savedSlots) {
        torch.runtime._debug_useSavedTensor(slot.record);
      }
      const unpackedTensors: Tensor[] = [];
      for (const slot of node.savedSlots) {
        // Default (non-hook) slots carry a graph-owned RuntimeTensor whose rc
        // keeps the saved value's storage alive independent of any scope. In
        // the common case the user's handle is still usable, so read through
        // it (backward math unchanged). If it was disposed / scope-reclaimed,
        // wrap the graph-owned tensor into a fresh handle so gradients stay
        // correct (the "disposing intermediates breaks autograd" fix). Hook
        // slots (checkpointing) recompute their value via unpackHook.
        const userHandle = slot.packed as Tensor;
        const handleUsable =
          !slot.retained ||
          (!!userHandle &&
            typeof userHandle._unwrap === "function" &&
            !userHandle.disposed);
        const tensor =
          handleUsable || !slot.retained
            ? slot.unpackHook(slot.packed)
            : torch._wrap(slot.retained, false);
        unpackedTensors.push(tensor);
      }
      unpacked.set(node, unpackedTensors);
    }
  } finally {
    torch._inCheckpointRecompute = false;
    // Close the recompute scope BEFORE any forcing. forceAllMerged creates
    // RuntimeTensors during plan execution that are normal outputs, NOT
    // checkpoint intermediates. Including them in the scope would cause
    // massive over-disposal in Full FT mode (588+ tensors/step).
    torch._checkpointRecomputeTensors = null;
    if (recomputeScope.size > 0) {
      _pendingCheckpointScopes.push(recomputeScope);
    }
  }

  return { unpacked, hasCheckpoints: recomputeScope.size > 0 };
}

/**
 * Run backward functions for all autograd nodes, accumulating gradients
 * into the gradMap.
 */
async function runBackwardFunctions(
  torch: Torchlette,
  ordered: AutogradNode[],
  gradMap: Map<Tensor, RuntimeTensor>,
  allUnpackedTensors: Map<AutogradNode, Tensor[]>,
): Promise<void> {
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
    } catch (_e: unknown) {
      torch.runtime.stopIntermediateTracking();
      await torch.runtime.force(gradOutTensor);
      torch.runtime.startIntermediateTracking();
      gradsIn = node.backward(gradOutTensor, getSaved);
    }

    const trackedIntermediates = torch.runtime.stopIntermediateTracking();

    const keepSet = new Set(
      gradsIn.filter((g): g is RuntimeTensor => g !== null),
    );
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
        // Old existing is tracked by TidyDispatchMode (not wrapped/escaped)
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
}

/**
 * Clean up saved tensors and autograd graph after backward.
 * Disposes forward intermediates and clears autograd node references.
 */
function cleanupAutogradGraph(
  torch: Torchlette,
  ordered: AutogradNode[],
  gradMap: Map<Tensor, RuntimeTensor>,
  deferForce = false,
): void {
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
    const isLeaf =
      tensor.requiresGrad && !tensor._gradNode() && !tensor.disposed;
    if (isLeaf) preserved.add(tensor);
  }
  for (const node of ordered) {
    for (const input of node.inputs) {
      // Preserve external user inputs (not forward intermediates, not
      // compile-created, and not already disposed by frontend tidy).
      if (forwardIntermediates.has(input)) continue;
      if (input.disposed) continue;
      if (torch._compileCreatedTensors.has(input)) continue;
      preserved.add(input);
    }
  }

  // Collect forward intermediates to dispose (excluding preserved tensors)
  const toDispose = new Set<Tensor>();
  for (const node of ordered) {
    if (!preserved.has(node.output)) {
      toDispose.add(node.output);
    }
  }

  // Collect saved slots and inputs from ordered nodes, PLUS any detached
  // autograd branches reachable from their inputs. Without this, detach()
  // makes upstream nodes unreachable from the loss's graph traversal, and
  // their saved tensors (kept by _wrapWithGrad) leak until GC.
  const visitedNodes = new Set<AutogradNode>();
  const processNode = (node: AutogradNode) => {
    if (visitedNodes.has(node)) return;
    visitedNodes.add(node);
    for (const slot of node.savedSlots) {
      // Release the graph-owned retention rc (symmetric with save time). This
      // is the SAME machinery as the compiled-plan harvest view-base retains —
      // every retain needs a release on every graph-teardown path. Under
      // whole-step trace the saved value still feeds the un-forced plan, so
      // defer the release to the boundary (after the single force).
      if (slot.retained && !slot.retained.disposed) {
        const retained = slot.retained;
        if (deferForce) {
          torch._deferToBoundary(() => {
            if (!retained.disposed) retained.dispose();
          });
        } else {
          retained.dispose();
        }
      }
      const savedTensor = slot.packed as Tensor;
      if (savedTensor && typeof savedTensor.dispose === "function") {
        if (!preserved.has(savedTensor) && !savedTensor.disposed) {
          toDispose.add(savedTensor);
        }
      }
    }
    for (const input of node.inputs) {
      if (!preserved.has(input) && !input.disposed) {
        toDispose.add(input);
      }
      // Walk into detached branches: if this input has its own autograd
      // chain that's NOT in ordered, recursively process it
      const inputNode = input._gradNode();
      if (inputNode && !visitedNodes.has(inputNode)) {
        processNode(inputNode);
      }
    }
    node.savedSlots.length = 0;
    node.output._setGradNode(null);
    node.inputs.length = 0;
  };
  for (const node of ordered) {
    processNode(node);
  }

  // Dispose all collected tensors — the forward intermediates + saved tensors.
  // Under whole-step trace these are inputs to the un-forced whole-step plan;
  // defer their disposal to the boundary drain (after the single force
  // consumes them). The autograd node structure was already torn down above,
  // so this only postpones freeing the tensor handles, not graph teardown.
  if (deferForce) {
    const deferred = [...toDispose];
    torch._deferToBoundary(() => {
      for (const tensor of deferred) {
        if (!tensor.disposed) tensor.dispose();
      }
    });
  } else {
    for (const tensor of toDispose) {
      if (!tensor.disposed) {
        tensor.dispose();
      }
    }
  }
}

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

  // [whole-step trace, P1] When active, this backward DEFERS its grad-write
  // force AND all teardown that assumes forcing happened, to the step boundary
  // (docs/step-function-compiler-design.md §3.1). Computed once, in scope for
  // the outer finally. GATED OFF for checkpointed backward below: P1 covers
  // NON-CHECKPOINT configs only — the structural recompute force + its
  // intermediate disposal (disposeCheckpointIntermediates) assume the grads
  // were already forced; deferring past them is a UAF. Checkpoint remat is
  // P3's pass (§3.3). `let` because hasCheckpoints is only known after
  // collectSavedTensors runs.
  let deferForce = torch._deferBackwardForce();
  return torch._runEntryPoint(async () => {
    // TidyDispatchMode tracks all RuntimeTensors created during backward
    // and auto-disposes non-escaped ones at scope exit. Leaf/retained
    // gradient RuntimeTensors are explicitly markEscaped so they survive.
    const tidyMode = new TidyDispatchMode();
    torch.runtime.pushDispatchMode(tidyMode);
    try {
      return await torch.runtime.runWithAsyncScope(async () => {
        const seed = grad ? grad._unwrap() : torch._seedGrad(a);
        const ownsSeed = !grad; // dispose seed after backward if we created it

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

        try {
        // Collect saved tensors (triggers checkpoint recomputation if any).
        const { unpacked: allUnpackedTensors, hasCheckpoints } =
          collectSavedTensors(torch, ordered);

        // [whole-step trace, P1] Checkpointed backward keeps the eager force:
        // the recompute force (autograd.ts, the hasCheckpoints branch) and
        // disposeCheckpointIntermediates() below both assume grads are forced
        // before teardown. Deferring past them frees recomputed inputs the
        // un-forced plan still reads (a UAF → GPU crash). Non-checkpoint only.
        if (hasCheckpoints) deferForce = false;

        if (hasCheckpoints) {
          // Checkpoint backward needs separate plans: forward tensors first,
          // then recomputed saved tensors. The saved-tensor plan must only
          // contain recomputed nodes — mixing in unmaterialized forward nodes
          // causes DSL rewrites to produce invalid reshape operations.
          const forwardToForce: RuntimeTensor[] = [];
          // Do NOT force-materialize the grad seed (`full([],1.0)`) in this
          // separate forward-tensors plan. The seed is a LEAF CONSTANT (no
          // inputs); forcing it here makes it a CROSS-PLAN value — produced in
          // this plan, consumed by the main backward plan (e.g. a GradScaler's
          // `mul(seed, scale)` backward). When the recorded build is retired and
          // the compiled plan is built from the generated stream, the
          // observed-liveness harvest cannot witness that later cross-plan read
          // (data-dependent — a GradScaler inf-skip re-fingerprints the plans),
          // so it prunes the seed's harvested `full` result; its rc hits 0 and
          // `destroyUnreachable` reaps it mid-backward, before the consumer reads
          // it (`[lifetime] reading RECLAIMED (shape=[])`). Leaving the seed lazy
          // materializes it INSIDE the main backward plan alongside its consumer
          // (intra-plan) — no cross-plan harvest, no prune, no reap. Backward
          // functions only build lazy nodes, so the recompute plan never needs
          // the seed's value early. Null on the recorded build (the harvest pins
          // the seed either way); the fix is for the recorded-build sunset.
          for (const node of ordered) {
            for (const input of node.inputs) {
              const rt = input._unwrap();
              if (!rt.disposed && !rt.isMaterialized()) {
                forwardToForce.push(rt);
              }
            }
          }
          if (forwardToForce.length > 0) {
            await torch.runtime.forceAllMerged(...forwardToForce);
          }

          // Force recomputed saved tensors.
          const savedToForce: RuntimeTensor[] = [];
          for (const tensors of allUnpackedTensors.values()) {
            for (const t of tensors) {
              savedToForce.push(t._unwrap());
            }
          }
          if (savedToForce.length > 0) {
            await torch.runtime.forceAllMerged(...savedToForce);
          }
        }
        // Non-checkpoint backward: skip both forces. All backward functions
        // only build lazy graph nodes (never read tensor values), so the
        // merged fwd+bwd plan reduces force points from 3 to 1.

        // Run backward functions
        await runBackwardFunctions(torch, ordered, gradMap, allUnpackedTensors);

        // Pre-compute leaf tensor set for gradient accumulation.
        const leafTensors = new Set<Tensor>();
        const retainTensors = new Set<Tensor>();
        for (const [tensor] of gradMap) {
          if (tensor.requiresGrad && !tensor._gradNode()) {
            leafTensors.add(tensor);
          }
          if (tensor.isRetainGrad) {
            retainTensors.add(tensor);
          }
        }

        // Dispose checkpoint-recomputed intermediates. The checkpoint owns
        // this — it tracked which tensors were recomputed (pending at capture
        // time) vs which were model weight references (already materialized).
        disposeCheckpointIntermediates();
        allUnpackedTensors.clear();

        // Drop non-leaf, non-retained grads from gradMap before force.
        // Their values did their accumulation work during runBackwardFunctions
        // and are now dead. Removing their live RuntimeTensor refs lets the
        // liveness-based early release reclaim their buffers inside the plan.
        //
        // A single RuntimeTensor can be the value for multiple keys (e.g.
        // add's backward returns dA = dB = dOut). Only dispose a grad if
        // NONE of its keys are leaves or retained.
        const survivorSet = new Set<RuntimeTensor>();
        for (const [tensor, gradRt] of gradMap) {
          if (leafTensors.has(tensor) || retainTensors.has(tensor)) {
            survivorSet.add(gradRt);
          }
        }
        // Under whole-step trace the non-survivor grads are inputs to the
        // un-forced optimizer plan — disposing them now reclaims their buffers
        // before the boundary force reads them. Keep them; the boundary sweep
        // reclaims the whole step's temporaries after the single force.
        if (!deferForce) {
          for (const [, gradRt] of gradMap) {
            if (!survivorSet.has(gradRt) && !gradRt.disposed) {
              gradRt.dispose();
            }
          }
        }

        // Cross-call gradient accumulation (PyTorch semantics): if a leaf (or
        // retained tensor) already carries a .grad from an EARLIER backward()
        // that wasn't zeroed, fold it into this pass's gradient — `.grad` is an
        // accumulator, exactly like torch.Tensor.grad (zeroGrad/zero_grad
        // resets it). Done BEFORE the force so the sum and BOTH its inputs (the
        // existing .grad + this pass's grad) materialize in the same merged
        // plan; the old .grad is disposed only later by _setGrad, after the sum
        // is concrete. Standard loops zero each iteration, so the existing grad
        // is null here and this is a no-op (accumulate ≡ overwrite) — the
        // change is invisible unless you intentionally skip zeroGrad (grad
        // accumulation across micro-batches).
        for (const tensor of gradMap.keys()) {
          if (!(leafTensors.has(tensor) || retainTensors.has(tensor))) continue;
          const existing = tensor.grad;
          if (!existing || existing.disposed) continue;
          const cur = gradMap.get(tensor);
          if (!cur || cur.disposed) continue;
          const summed = torch.runtime.add(existing._unwrap(), cur);
          gradMap.set(tensor, summed);
          // `cur` stays alive (input to `summed`); just hand the survivor slot
          // to `summed` so it (not the un-summed grad) gets forced + written.
          survivorSet.delete(cur);
          survivorSet.add(summed);
        }

        // Force the surviving gradients in one merged plan — UNLESS the
        // whole-step trace scope is active (P1,
        // docs/step-function-compiler-design.md §3.1). Deferring this force
        // leaves the grads lazy so the optimizer builds its update graph on
        // top of them and the step boundary (endStep/markStep forceAllPending,
        // or the capture ring's `_deferBoundaryCommit`) forces forward +
        // backward + optimizer as ONE plan — the census's DEFERRED-LOSS config
        // taken one force further (grads no longer force separately from the
        // optimizer). The saved-for-backward inputs of these un-forced grad
        // nodes stay alive by graph edge exactly as the forward activations
        // already do in the deferred-loss (merged fwd+bwd) plan — this simply
        // extends that same edge-liveness one segment further, to the boundary.
        // The post-force `destroyUnreachable` is deferred with it: the boundary
        // sweep reclaims the whole step's temporaries. Reads of a grad before
        // the boundary still materialize on demand (correctness-preserving);
        // deferral changes only WHEN the force happens.
        if (!deferForce) {
          // Force the surviving gradients in one merged plan.
          const allGrads = [...survivorSet].filter(
            (g) => !g.isMaterialized() && !g.disposed,
          );
          if (allGrads.length > 0) {
            await torch.runtime.forceAllMerged(...allGrads);
          }

          storageTracker.destroyUnreachable();
        }

        // Store final gradients on leaf tensors and retained non-leaf tensors
        // Mark these tensors as "kept" so they survive async scope cleanup
        for (const [tensor, gradTensor] of gradMap) {
          if (leafTensors.has(tensor) || retainTensors.has(tensor)) {
            // Wrap and keep the gradient tensor before async scope exits
            const gradWrapper = torch._wrap(gradTensor, false);
            torch.keep(gradWrapper);
            tensor._setGrad(gradWrapper);
          }
        }

        cleanupAutogradGraph(torch, ordered, gradMap, deferForce);

        // Dispose the gradient seed if we created it. Like TF.js, don't rely
        // on tidy to catch it — explicitly dispose consumed intermediates.
        // Under whole-step trace the seed feeds the un-forced backward plan;
        // defer its disposal to the boundary.
        if (ownsSeed) {
          if (deferForce) torch._deferToBoundary(() => seed.dispose());
          else seed.dispose();
        }
        } finally {
          // Symmetric release of graph-owned retention rcs on EVERY exit,
          // including an error thrown before cleanupAutogradGraph. On the
          // success path cleanupAutogradGraph already cleared savedSlots, so
          // this loop is a no-op; on the error path it prevents a leaked rc
          // (a leaked retain pins GPU memory forever).
          for (const node of ordered) {
            for (const slot of node.savedSlots) {
              if (slot.retained && !slot.retained.disposed) {
                slot.retained.dispose();
              }
            }
          }
        }
      });
    } finally {
      torch.runtime.popDispatchMode();
      // Under whole-step trace the backward grad-chain intermediates (tracked,
      // non-escaped, in `tidyMode`) are inputs to the un-forced optimizer plan;
      // disposing them here reclaims their buffers before the boundary force.
      // Defer the sweep to the boundary drain (after the single force). On the
      // eager path this stays exactly as before.
      if (deferForce) {
        torch._deferToBoundary(() => {
          tidyMode.disposeNonEscaped();
          storageTracker.destroyUnreachable();
        });
      } else {
        tidyMode.disposeNonEscaped();
        storageTracker.destroyUnreachable();
      }
    }
  });
}
