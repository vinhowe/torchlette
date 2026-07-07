# Lazy Execution Design

## Overview

Per the working spec (§0.1, §1.2, §3.4), all operations are globally lazy:
- Ops return immediately without executing device work
- Device work is delayed until a force boundary (cpu(), item(), markStep())
- Outside compiled regions: no optimization, just sequential execution

## Current vs Target Architecture

### Current (Eager)
```
Tensor.add(a, b)
    │
    ├──► Engine.afterAll()     → tracks tokens (semantic only)
    │
    └──► RuntimeEngine.add()   → backend.ops.add() → EXECUTES IMMEDIATELY
```

### Target (Lazy)
```
Tensor.add(a, b)
    │
    └──► LazyEngine.add()
            │
            ├──► Creates IRNode(op="add", inputs=[a.lazyRef, b.lazyRef])
            ├──► Updates tokens
            └──► Returns Tensor with LazyRef (NO EXECUTION)

Tensor.cpu()  (force boundary)
    │
    └──► LazyEngine.force(tensor.lazyRef)
            │
            ├──► Collects reachable IRNodes (dependency closure)
            ├──► Topological sort → execution plan
            ├──► For each node: backend.ops.X() → materialize
            ├──► Commit versions on success
            └──► Return materialized value
```

## Core Types

```typescript
// Storage handle for materialized values
interface StorageHandle {
  id: number;
  device: DeviceKind;
  backendTensor: BackendTensor;  // actual data
}

// Lazy reference - either pending computation or materialized
type LazyRef =
  | { kind: "pending"; node: IRNode }
  | { kind: "materialized"; storage: StorageHandle };

// IR Node representing a pending operation
interface IRNode {
  id: number;
  op: OpCode;
  inputs: LazyRef[];
  shape: number[];
  dtype: DType;
  device: DeviceKind;

  // For effect ordering (§3.1-3.3)
  tokenIn?: Token;
  tokenOut?: Token;

  // Cached result after execution
  result?: StorageHandle;
}

// Extended base binding (§1.1)
interface BaseBinding {
  kind: "ssa" | "loc" | "pending_loc";

  // SSA-backed: the lazy value
  value?: LazyRef;

  // Loc-backed: which loc holds the value
  locId?: LocId;

  // Pending-loc: initialization token
  initTok?: Token;
}
```

## Execution Flow

### 1. Op Emission (Lazy)
```typescript
add(a: Tensor, b: Tensor): Tensor {
  // Get lazy refs for inputs
  const aRef = this.resolveBaseValue(a.baseId);
  const bRef = this.resolveBaseValue(b.baseId);

  // Infer output shape/dtype
  const shape = broadcastShapes(a.shape, b.shape);
  const dtype = a.dtype;

  // Create IR node (NO EXECUTION)
  const node: IRNode = {
    id: this.nextNodeId++,
    op: "add",
    inputs: [aRef, bRef],
    shape,
    dtype,
    device: a.device,
  };

  // Create lazy ref
  const lazyRef: LazyRef = { kind: "pending", node };

  // Allocate new BaseId with SSA binding
  const baseId = this.allocBaseId();
  this.baseBindings.set(baseId, { kind: "ssa", value: lazyRef });

  // Return tensor handle
  return new Tensor(baseId, shape, dtype, device);
}
```

### 2. Force Boundary
```typescript
async force(baseId: BaseId): Promise<StorageHandle> {
  const binding = this.baseBindings.get(baseId);
  const lazyRef = this.resolveBaseValue(baseId);

  if (lazyRef.kind === "materialized") {
    return lazyRef.storage;  // Already computed
  }

  // Build execution plan
  const plan = this.buildPlan(lazyRef.node);

  // Execute plan
  const result = await this.executePlan(plan);

  // Commit versions on success
  this.commitPlanVersions(plan);

  return result;
}

buildPlan(root: IRNode): ExecutionPlan {
  // Collect all reachable pending nodes
  const nodes: IRNode[] = [];
  const visited = new Set<number>();

  const visit = (ref: LazyRef) => {
    if (ref.kind === "materialized") return;
    if (visited.has(ref.node.id)) return;
    visited.add(ref.node.id);

    // Visit inputs first (dependencies)
    for (const input of ref.node.inputs) {
      visit(input);
    }
    nodes.push(ref.node);
  };

  visit({ kind: "pending", node: root });

  // nodes is now in topological order
  return { nodes };
}

async executePlan(plan: ExecutionPlan): Promise<StorageHandle> {
  for (const node of plan.nodes) {
    // Get materialized inputs
    const inputs = node.inputs.map(ref => {
      if (ref.kind === "materialized") return ref.storage;
      if (ref.node.result) return ref.node.result;
      throw new Error("Input not yet computed");
    });

    // Execute via backend
    const result = this.backend.ops[node.op](...inputs.map(s => s.backendTensor));

    // Cache result in node
    node.result = {
      id: this.nextStorageId++,
      device: node.device,
      backendTensor: result,
    };
  }

  return plan.nodes[plan.nodes.length - 1].result!;
}
```

### 3. markStep() (§6.1)
```typescript
async markStep(): Promise<void> {
  // Force entire tokGlobal stream
  // This materializes all pending effects

  // Collect all pending nodes reachable from tokGlobal
  const allPending = this.collectTokenReachable(this.tokGlobal);

  if (allPending.length > 0) {
    const plan = this.buildPlanFromNodes(allPending);
    await this.executePlan(plan);
    this.commitPlanVersions(plan);
  }

  // Reset tokens (§6.6)
  this.tokGlobal = this.tokenStore.root;
  this.tokLoc.clear();
}
```

## Implementation Phases

### Phase 1: LazyRef and IRNode ✅ COMPLETE
- Define LazyRef, LazyIRNode types in src/engine/lazy.ts
- Create helper functions: createPendingRef, createMaterializedRef, isPending, isMaterialized
- Tests: 18 tests for lazy refs, node creation, plan building

### Phase 2: Lazy Op Emission ✅ COMPLETE
- LazyIRNode stores op, inputs, shape, dtype, device, payload
- buildPlan() collects reachable nodes in topological order
- Tests: 12 tests for shape/dtype/device tracking, payload handling

### Phase 3: Force and Execute ✅ COMPLETE
- executePlan() dispatches nodes through backend ops
- Supports all ops: add, sub, mul, div, matmul, sqrt, relu, reshape, transpose, etc.
- Caches results in node.result for reuse
- Tests: 14 tests for execution, diamond DAG, error handling

### Phase 4: Version Commits (NEXT)
- Implement commitPlanVersions()
- Handle execution failure (poison engine)
- Tests: versions update correctly

### Phase 5: Frontend Integration (NEXT)
- Create LazyEngine that wraps semantic Engine + lazy execution
- Modify frontend Tensor to hold LazyRef instead of BackendTensor
- Force at cpu(), item(), markStep() boundaries
- Ensure existing tests pass

## Files to Modify/Create

**New:**
- `src/engine/lazy.ts` - LazyRef, IRNode, lazy op helpers
- `src/engine/executor.ts` - Plan building and execution
- `test/lazy-execution.spec.ts` - Lazy execution tests

**Modify:**
- `src/engine/engine.ts` - Integrate lazy execution
- `src/frontend.ts` - Use lazy engine
- `src/runtime/engine.ts` - May be deprecated or repurposed as executor
