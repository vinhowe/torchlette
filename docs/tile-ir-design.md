# Tile IR: Triton-like Fusion Architecture

## Executive Summary

This document proposes a unified tile-based intermediate representation (IR) for kernel fusion in Torchlette's WebGPU backend. The goal is to eliminate duplicate op definitions, enable arbitrary fusion patterns, and provide a foundation for more sophisticated optimizations.

**Current Pain Points:**
- Op expressions defined in 3 places (fusion-codegen, epilogue, matmul codegen)
- Matmul epilogue only supports 6 ops vs 30+ in elementwise fusion
- No unified representation for "a fused kernel"
- Hard to extend with new fusion patterns (e.g., reduction epilogues)

**Proposed Solution:**
- Single op registry with WGSL expression generators
- Tile-based IR where all operations work on register/shared-memory tiles
- Unified codegen path for all fusion patterns

---

## Phase 1: Unified Op Registry

**Effort: 2-3 hours**

### Current State

```
src/backend/webgpu/fusion-codegen.ts
├── UNARY_EXPR: Record<string, (input) => string>      # 20+ ops
├── BINARY_EXPR: Record<string, (a, b) => string>      # 15+ ops
└── isFusibleOp(), getExprGenerator()

src/backend/webgpu/matmul/epilogue.ts
├── FUSIBLE_UNARY_OPS: Set<string>                     # 10 ops (subset)
├── FUSIBLE_BINARY_OPS: Set<string>                    # 4 ops (subset)
└── isFusibleIntoEpilogue()

src/backend/webgpu/matmul/codegen.ts
├── EpilogueOp union type                              # 8 variants
└── genEpilogueCode() switch statement                 # inline WGSL generation
```

### Target State

```
src/backend/webgpu/ops/registry.ts      # NEW: Single source of truth
├── OpDef type
├── OP_REGISTRY: Record<string, OpDef>
├── getExpr(op, inputs): string
├── isFusible(op): boolean
└── canVectorize(op): boolean

src/backend/webgpu/fusion-codegen.ts    # REFACTOR: imports from registry
└── Uses OP_REGISTRY.getExpr()

src/backend/webgpu/matmul/codegen.ts    # REFACTOR: imports from registry
└── genEpilogueCode() uses OP_REGISTRY.getExpr()
```

### Op Registry Design

```typescript
// src/backend/webgpu/ops/registry.ts

export type OpArity = 1 | 2 | 3;

export interface OpDef {
  /** WGSL expression generator */
  expr: (...inputs: string[]) => string;

  /** Number of inputs */
  arity: OpArity;

  /** Can this op be fused into elementwise kernels? */
  fusible: boolean;

  /** Can this op be vectorized (vec2/vec4)? */
  vectorizable: boolean;

  /** Does this op need special handling for vector zero/one constants? */
  needsVectorConstants?: boolean;

  /** Output dtype (if different from input, e.g., comparisons return f32 0/1) */
  outputDtype?: 'same' | 'f32' | 'bool';
}

export const OP_REGISTRY: Record<string, OpDef> = {
  // Activations
  relu: {
    expr: (a, zero = "0.0") => `select(${zero}, ${a}, ${a} > ${zero})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    needsVectorConstants: true,
  },
  gelu: {
    expr: (a) => `(${a} * 0.5 * (1.0 + tanh(clamp(0.7978845608 * (${a} + 0.044715 * ${a} * ${a} * ${a}), -10.0, 10.0))))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
  },
  silu: {
    expr: (a) => `(${a} / (1.0 + exp(-${a})))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
  },
  sigmoid: {
    expr: (a) => `(1.0 / (1.0 + exp(-${a})))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
  },
  tanh: {
    expr: (a) => `tanh(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
  },
  softplus: {
    expr: (a) => `log(1.0 + exp(${a}))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
  },

  // Math functions
  neg: { expr: (a) => `(-${a})`, arity: 1, fusible: true, vectorizable: true },
  abs: { expr: (a) => `abs(${a})`, arity: 1, fusible: true, vectorizable: true },
  exp: { expr: (a) => `exp(${a})`, arity: 1, fusible: true, vectorizable: true },
  log: { expr: (a) => `log(${a})`, arity: 1, fusible: true, vectorizable: true },
  sqrt: { expr: (a) => `sqrt(${a})`, arity: 1, fusible: true, vectorizable: true },
  rsqrt: { expr: (a) => `inverseSqrt(${a})`, arity: 1, fusible: true, vectorizable: true },
  sin: { expr: (a) => `sin(${a})`, arity: 1, fusible: true, vectorizable: true },
  cos: { expr: (a) => `cos(${a})`, arity: 1, fusible: true, vectorizable: true },
  floor: { expr: (a) => `floor(${a})`, arity: 1, fusible: true, vectorizable: true },
  ceil: { expr: (a) => `ceil(${a})`, arity: 1, fusible: true, vectorizable: true },
  round: { expr: (a) => `round(${a})`, arity: 1, fusible: true, vectorizable: true },
  sign: { expr: (a) => `sign(${a})`, arity: 1, fusible: true, vectorizable: true },

  // Binary arithmetic
  add: { expr: (a, b) => `(${a} + ${b})`, arity: 2, fusible: true, vectorizable: true },
  sub: { expr: (a, b) => `(${a} - ${b})`, arity: 2, fusible: true, vectorizable: true },
  mul: { expr: (a, b) => `(${a} * ${b})`, arity: 2, fusible: true, vectorizable: true },
  div: { expr: (a, b) => `(${a} / ${b})`, arity: 2, fusible: true, vectorizable: true },
  pow: { expr: (a, b) => `pow(${a}, ${b})`, arity: 2, fusible: true, vectorizable: true },
  min: { expr: (a, b) => `min(${a}, ${b})`, arity: 2, fusible: true, vectorizable: true },
  max: { expr: (a, b) => `max(${a}, ${b})`, arity: 2, fusible: true, vectorizable: true },
  mod: { expr: (a, b) => `(${a} % ${b})`, arity: 2, fusible: true, vectorizable: false },

  // Comparisons (return 0.0 or 1.0)
  eq: { expr: (a, b) => `select(0.0, 1.0, ${a} == ${b})`, arity: 2, fusible: true, vectorizable: true, outputDtype: 'f32' },
  ne: { expr: (a, b) => `select(0.0, 1.0, ${a} != ${b})`, arity: 2, fusible: true, vectorizable: true, outputDtype: 'f32' },
  lt: { expr: (a, b) => `select(0.0, 1.0, ${a} < ${b})`, arity: 2, fusible: true, vectorizable: true, outputDtype: 'f32' },
  le: { expr: (a, b) => `select(0.0, 1.0, ${a} <= ${b})`, arity: 2, fusible: true, vectorizable: true, outputDtype: 'f32' },
  gt: { expr: (a, b) => `select(0.0, 1.0, ${a} > ${b})`, arity: 2, fusible: true, vectorizable: true, outputDtype: 'f32' },
  ge: { expr: (a, b) => `select(0.0, 1.0, ${a} >= ${b})`, arity: 2, fusible: true, vectorizable: true, outputDtype: 'f32' },

  // Ternary
  where: {
    expr: (cond, a, b) => `select(${b}, ${a}, ${cond} > 0.0)`,
    arity: 3,
    fusible: true,
    vectorizable: true
  },

  // Casts
  cast_f16: { expr: (a) => `f16(${a})`, arity: 1, fusible: true, vectorizable: false },
  cast_f32: { expr: (a) => `f32(${a})`, arity: 1, fusible: true, vectorizable: false },
  cast_i32: { expr: (a) => `i32(${a})`, arity: 1, fusible: true, vectorizable: false },
  cast_u32: { expr: (a) => `u32(${a})`, arity: 1, fusible: true, vectorizable: false },
};

/** Get expression for an op */
export function getExpr(op: string, inputs: string[], vectorConstants?: { zero: string; one: string }): string {
  const def = OP_REGISTRY[op];
  if (!def) throw new Error(`Unknown op: ${op}`);

  if (def.needsVectorConstants && vectorConstants) {
    return def.expr(inputs[0], vectorConstants.zero, vectorConstants.one);
  }
  return def.expr(...inputs);
}

/** Check if op can be fused */
export function isFusible(op: string): boolean {
  return OP_REGISTRY[op]?.fusible ?? false;
}

/** Check if op can be vectorized */
export function canVectorize(op: string): boolean {
  return OP_REGISTRY[op]?.vectorizable ?? false;
}

/** Get op arity */
export function getArity(op: string): OpArity {
  return OP_REGISTRY[op]?.arity ?? 1;
}
```

### Migration Steps

1. Create `src/backend/webgpu/ops/registry.ts` with the above
2. Update `fusion-codegen.ts`:
   - Import from registry
   - Replace `UNARY_EXPR`/`BINARY_EXPR` with calls to `getExpr()`
3. Update `matmul/epilogue.ts`:
   - Remove `FUSIBLE_UNARY_OPS`/`FUSIBLE_BINARY_OPS`
   - Use `isFusible()` from registry
4. Update `matmul/codegen.ts`:
   - Simplify `EpilogueOp` to just `{ op: string, inputs?: number[] }`
   - Replace switch statement with `getExpr()` call
5. Update tests

---

## Phase 2: Tile IR Design

**Effort: 1-2 days**

### Motivation

Current IR is per-element. Fusion is detected via pattern matching on the IR graph. This works but:
- Pattern matching is fragile
- Hard to express "compute this tile, then apply these ops"
- No clear boundary between "memory" and "compute"

Tile IR makes the memory/compute boundary explicit:
- **Load**: Memory → Tile (in registers or shared memory)
- **Compute**: Tile → Tile (no memory access)
- **Store**: Tile → Memory

### Tile IR Types

```typescript
// src/backend/webgpu/tile-ir/types.ts

/** A tile is a rectangular block of data in registers or shared memory */
export type TileLocation = 'register' | 'shared';

export type TileShape = {
  dims: number[];      // [M, N] for 2D, [M, N, K] for 3D
  dtype: DType;
  location: TileLocation;
};

/** Unique ID for a tile value in the IR */
export type TileId = number;

/** Memory access pattern */
export type MemoryLayout = {
  basePtr: number;     // Which input buffer
  shape: number[];     // Logical shape
  strides: number[];   // Memory strides
  offset: number;      // Base offset
};

/** Tile IR operations */
export type TileOp =
  // Memory operations
  | {
      op: 'tile_load';
      result: TileId;
      layout: MemoryLayout;
      tileShape: TileShape;
      // Coordinates are computed from workgroup/thread IDs
    }
  | {
      op: 'tile_store';
      tile: TileId;
      layout: MemoryLayout;
    }

  // Compute operations
  | {
      op: 'tile_matmul';
      result: TileId;
      a: TileId;
      b: TileId;
      // Accumulates: result += a @ b
      accumulate?: TileId;
    }
  | {
      op: 'tile_reduce';
      result: TileId;
      input: TileId;
      dim: number;
      reduceOp: 'sum' | 'max' | 'min' | 'prod';
    }
  | {
      op: 'tile_broadcast';
      result: TileId;
      input: TileId;
      targetShape: number[];
    }
  | {
      op: 'tile_elementwise';
      result: TileId;
      inputs: TileId[];
      expr: string;        // Op name from registry
    }

  // Control flow
  | {
      op: 'tile_loop';
      variable: string;
      start: number;
      end: number;
      step: number;
      body: TileOp[];
    }
  | {
      op: 'tile_sync';
      // Workgroup barrier
    };

/** A complete tile kernel */
export interface TileKernel {
  name: string;

  // Input/output buffers
  inputs: {
    name: string;
    dtype: DType;
    shape: number[];
  }[];
  outputs: {
    name: string;
    dtype: DType;
    shape: number[];
  }[];

  // Workgroup configuration
  workgroupSize: [number, number, number];

  // The ops to execute
  ops: TileOp[];

  // Metadata for code generation
  sharedMemoryBytes: number;
  registersPerThread: number;
}
```

### Example: Matmul + ReLU + Bias

**Current LazyIR:**
```
node1: matmul(A, B)
node2: relu(node1)
node3: add(node2, bias)
```

**Tile IR:**
```typescript
{
  name: "matmul_relu_bias",
  inputs: [
    { name: "A", dtype: "f32", shape: [M, K] },
    { name: "B", dtype: "f32", shape: [K, N] },
    { name: "bias", dtype: "f32", shape: [N] },
  ],
  outputs: [
    { name: "C", dtype: "f32", shape: [M, N] },
  ],
  workgroupSize: [16, 16, 1],
  ops: [
    // Accumulator tile (in registers)
    { op: 'tile_elementwise', result: 0, inputs: [], expr: 'zero' },  // acc = 0

    // Loop over K tiles
    {
      op: 'tile_loop',
      variable: 'k_tile',
      start: 0,
      end: K,
      step: TILE_K,
      body: [
        // Load A tile to shared memory
        { op: 'tile_load', result: 1, layout: { basePtr: 0, ... }, tileShape: { dims: [TILE_M, TILE_K], location: 'shared' } },
        // Load B tile to shared memory
        { op: 'tile_load', result: 2, layout: { basePtr: 1, ... }, tileShape: { dims: [TILE_K, TILE_N], location: 'shared' } },
        // Sync
        { op: 'tile_sync' },
        // Matmul accumulate
        { op: 'tile_matmul', result: 0, a: 1, b: 2, accumulate: 0 },
        // Sync before next iteration
        { op: 'tile_sync' },
      ]
    },

    // Epilogue: relu
    { op: 'tile_elementwise', result: 3, inputs: [0], expr: 'relu' },

    // Epilogue: add bias (broadcast load)
    { op: 'tile_load', result: 4, layout: { basePtr: 2, ... }, tileShape: { dims: [1, TILE_N], location: 'register' } },
    { op: 'tile_broadcast', result: 5, input: 4, targetShape: [TILE_M, TILE_N] },
    { op: 'tile_elementwise', result: 6, inputs: [3, 5], expr: 'add' },

    // Store result
    { op: 'tile_store', tile: 6, layout: { basePtr: 'output', ... } },
  ],
  sharedMemoryBytes: (TILE_M * TILE_K + TILE_K * TILE_N) * 4,
  registersPerThread: THREAD_TILE_M * THREAD_TILE_N,
}
```

### Example: Elementwise Fusion

**Current LazyIR:**
```
node1: add(A, B)
node2: relu(node1)
node3: mul(node2, C)
```

**Tile IR:**
```typescript
{
  name: "add_relu_mul",
  inputs: [
    { name: "A", dtype: "f32", shape: [M, N] },
    { name: "B", dtype: "f32", shape: [M, N] },
    { name: "C", dtype: "f32", shape: [M, N] },
  ],
  outputs: [
    { name: "out", dtype: "f32", shape: [M, N] },
  ],
  workgroupSize: [256, 1, 1],
  ops: [
    // Load tiles (vectorized)
    { op: 'tile_load', result: 0, layout: { basePtr: 0, ... }, tileShape: { dims: [4], location: 'register' } },
    { op: 'tile_load', result: 1, layout: { basePtr: 1, ... }, tileShape: { dims: [4], location: 'register' } },
    { op: 'tile_load', result: 2, layout: { basePtr: 2, ... }, tileShape: { dims: [4], location: 'register' } },

    // Compute
    { op: 'tile_elementwise', result: 3, inputs: [0, 1], expr: 'add' },
    { op: 'tile_elementwise', result: 4, inputs: [3], expr: 'relu' },
    { op: 'tile_elementwise', result: 5, inputs: [4, 2], expr: 'mul' },

    // Store
    { op: 'tile_store', tile: 5, layout: { basePtr: 'output', ... } },
  ],
  sharedMemoryBytes: 0,
  registersPerThread: 4 * 3,  // 3 input vec4s
}
```

---

## Phase 3: LazyIR → TileIR Lowering

**Effort: 2-3 days**

### Overview

This phase converts our existing LazyIR (per-element operations) into TileIR (tile operations). The key decisions:

1. **Fusion boundary detection**: Which ops fuse together?
2. **Tiling strategy**: What tile sizes to use?
3. **Memory/compute classification**: Which ops are loads vs compute?

### Fusion Region Detection

```typescript
// src/backend/webgpu/tile-ir/fusion-regions.ts

export interface FusionRegion {
  /** IDs of nodes in this region */
  nodeIds: Set<number>;

  /** The "anchor" op that determines tiling (matmul, reduction, or elementwise) */
  anchor: 'matmul' | 'reduction' | 'elementwise';

  /** External inputs (nodes not in region) */
  externalInputs: number[];

  /** Outputs consumed outside region */
  outputs: number[];
}

/**
 * Detect fusion regions in an IR graph.
 *
 * Strategy:
 * 1. Find "anchor" ops (matmul, reduction) - these set the tiling
 * 2. Grow regions forward through fusible elementwise ops
 * 3. Grow regions backward through fusible elementwise ops
 * 4. Remaining elementwise chains form their own regions
 */
export function detectFusionRegions(graph: IRGraph): FusionRegion[] {
  // Implementation
}
```

### Tiling Strategy

```typescript
// src/backend/webgpu/tile-ir/tiling.ts

export interface TilingConfig {
  // For matmul anchor
  tileM: number;
  tileN: number;
  tileK: number;
  threadTileM: number;
  threadTileN: number;

  // For reduction anchor
  reduceBlockSize: number;

  // For elementwise anchor
  vectorWidth: 1 | 2 | 4;
  elementsPerThread: number;
}

/**
 * Select tiling parameters based on:
 * - Problem size (M, N, K)
 * - Available shared memory
 * - Register pressure
 * - Hardware capabilities (subgroups?)
 */
export function selectTiling(
  region: FusionRegion,
  graph: IRGraph,
  deviceLimits: GPULimits,
): TilingConfig {
  // Implementation - can reuse existing autotuning logic
}
```

### Lowering Pass

```typescript
// src/backend/webgpu/tile-ir/lower.ts

/**
 * Lower a fusion region to TileIR.
 */
export function lowerToTileIR(
  region: FusionRegion,
  graph: IRGraph,
  tiling: TilingConfig,
): TileKernel {
  const anchor = findAnchorOp(region, graph);

  switch (anchor.type) {
    case 'matmul':
      return lowerMatmulRegion(region, graph, tiling);
    case 'reduction':
      return lowerReductionRegion(region, graph, tiling);
    case 'elementwise':
      return lowerElementwiseRegion(region, graph, tiling);
  }
}

function lowerMatmulRegion(
  region: FusionRegion,
  graph: IRGraph,
  tiling: TilingConfig,
): TileKernel {
  const ops: TileOp[] = [];

  // 1. Initialize accumulator
  ops.push({ op: 'tile_elementwise', result: nextId(), inputs: [], expr: 'zero' });

  // 2. K-loop with tile loads and matmul
  ops.push({
    op: 'tile_loop',
    variable: 'k',
    start: 0,
    end: K,
    step: tiling.tileK,
    body: generateMatmulLoopBody(tiling),
  });

  // 3. Epilogue ops (everything after matmul in the region)
  const epilogueOps = getEpilogueOps(region, graph);
  for (const node of epilogueOps) {
    ops.push({
      op: 'tile_elementwise',
      result: nextId(),
      inputs: mapInputs(node, region),
      expr: node.op,  // Uses op registry
    });
  }

  // 4. Store
  ops.push({ op: 'tile_store', tile: lastResult(), layout: outputLayout });

  return { name: generateName(region), ops, ... };
}
```

---

## Phase 4: TileIR → WGSL Codegen

**Effort: 2-3 days**

### Overview

Generate WGSL compute shaders from TileIR. The key insight: most complexity is in the lowering (Phase 3). Codegen is relatively mechanical.

### Code Generator

```typescript
// src/backend/webgpu/tile-ir/codegen.ts

export function generateWGSL(kernel: TileKernel): string {
  const ctx = new CodegenContext(kernel);

  // 1. Generate bindings
  const bindings = generateBindings(kernel);

  // 2. Generate shared memory declarations
  const sharedDecls = generateSharedDecls(kernel);

  // 3. Generate main function
  const mainBody = generateOps(kernel.ops, ctx);

  return `
${bindings}

${sharedDecls}

@compute @workgroup_size(${kernel.workgroupSize.join(', ')})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
${mainBody}
}
`;
}

class CodegenContext {
  private tileVars = new Map<TileId, string>();
  private nextVar = 0;

  getTileVar(id: TileId): string {
    if (!this.tileVars.has(id)) {
      this.tileVars.set(id, `t${this.nextVar++}`);
    }
    return this.tileVars.get(id)!;
  }
}

function generateOps(ops: TileOp[], ctx: CodegenContext): string {
  const lines: string[] = [];

  for (const op of ops) {
    switch (op.op) {
      case 'tile_load':
        lines.push(generateTileLoad(op, ctx));
        break;
      case 'tile_store':
        lines.push(generateTileStore(op, ctx));
        break;
      case 'tile_matmul':
        lines.push(generateTileMatmul(op, ctx));
        break;
      case 'tile_elementwise':
        lines.push(generateTileElementwise(op, ctx));
        break;
      case 'tile_loop':
        lines.push(generateTileLoop(op, ctx));
        break;
      case 'tile_sync':
        lines.push('workgroupBarrier();');
        break;
    }
  }

  return lines.join('\n');
}

function generateTileElementwise(op: TileElementwise, ctx: CodegenContext): string {
  const resultVar = ctx.getTileVar(op.result);
  const inputVars = op.inputs.map(id => ctx.getTileVar(id));

  // Use the unified op registry!
  const expr = getExpr(op.expr, inputVars);

  return `let ${resultVar} = ${expr};`;
}
```

### Template Library

For complex patterns like matmul, we can use templates:

```typescript
// src/backend/webgpu/tile-ir/templates/matmul.ts

export function generateTileMatmul(
  op: TileMatmulOp,
  ctx: CodegenContext,
  tiling: TilingConfig,
): string {
  // This is essentially our current matmul codegen,
  // but now it's a modular piece that TileIR composes
  return `
  // Thread tile computation
  for (var i = 0u; i < ${tiling.threadTileM}u; i++) {
    for (var j = 0u; j < ${tiling.threadTileN}u; j++) {
      for (var k = 0u; k < TILE_K; k++) {
        ${ctx.getTileVar(op.result)}[i * ${tiling.threadTileN} + j] +=
          tileA[...] * tileB[...];
      }
    }
  }
  `;
}
```

---

## Implementation Roadmap

### Week 1: Phase 1 (Foundation)
- [ ] Create `src/backend/webgpu/ops/registry.ts`
- [ ] Migrate `fusion-codegen.ts` to use registry
- [ ] Migrate `matmul/codegen.ts` to use registry
- [ ] Update tests
- [ ] Verify all existing tests pass

### Week 2: Phase 2 (Design)
- [ ] Create `src/backend/webgpu/tile-ir/types.ts`
- [ ] Write TileIR examples for key patterns
- [ ] Design fusion region detection algorithm
- [ ] Document tiling strategy

### Week 3: Phase 3 (Lowering)
- [ ] Implement fusion region detection
- [ ] Implement elementwise lowering
- [ ] Implement matmul lowering
- [ ] Integration tests

### Week 4: Phase 4 (Codegen)
- [ ] Implement WGSL codegen for TileIR
- [ ] Port matmul templates
- [ ] End-to-end tests
- [ ] Performance benchmarks

---

## Future Extensions

Once TileIR is in place, several optimizations become easier:

### Multi-Kernel Fusion
```
matmul1 → layernorm → matmul2
```
Could be expressed as a single TileKernel with intermediate tiles staying in shared memory.

### Reduction Epilogues
```
sum(relu(x))  →  TileIR: load → relu → reduce
```
Currently not supported; TileIR makes this natural.

### Flash Attention
The flash attention algorithm (tiled softmax + matmul) maps directly to TileIR:
```
for each K tile:
  load Q tile, K tile
  compute QK^T tile (matmul)
  compute softmax (reduction + elementwise)
  load V tile
  compute output tile (matmul)
  accumulate with running max correction
```

### Autotuning
TileIR provides a clean interface for autotuning:
- Vary tiling parameters
- Generate kernel variants
- Benchmark and select best

---

## Appendix: Comparison with Other Systems

| System | Fusion Model | Tiling | Codegen |
|--------|-------------|--------|---------|
| PyTorch Inductor | Pattern matching → Triton | Triton decides | Triton |
| JAX/XLA | HLO fusion passes | XLA decides | LLVM/custom |
| Triton | Explicit in user code | User specifies | Triton compiler |
| TVM | Relay → TE → TIR | Schedule primitives | TIR → target |
| **Torchlette (proposed)** | LazyIR → TileIR | Autotuned | TileIR → WGSL |

Our approach is closest to TVM's, but simpler:
- We don't have a scheduling language (tiling is automatic)
- We generate WGSL directly (no LLVM)
- TileIR is higher-level than TVM's TIR
