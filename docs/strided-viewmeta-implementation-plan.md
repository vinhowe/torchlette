> **SUPERSEDED (2026-07-05, task #58).** This draft's diagnosis predated the fix:
> stride support already existed tree-wide; the actual bug was the strides-only
> `isContiguous` predicate trusted by ~20 raw-bind consumers. Fixed in acf851a0
> (isRawBindable + assertRawBindable + per-family offset handling). Kept for history.

# Strided ViewMeta Implementation Plan

This document describes the full implementation required to support strided views per §4.2-4.4 of the working spec.

## Current State

### What Exists
- **BaseId propagation**: Views correctly share baseId with their source
- **CPU stride support**: `readAtLinear()` and `writeAtLinear()` handle strided access
- **Transpose strides**: `transpose()` swaps strides without copying data
- **Contiguity check**: `isContiguous()` function exists in CPU backend

### What's Missing
- **ViewMeta type**: No structured metadata for views
- **IR-level tracking**: LazyIRNode/IRNode don't carry stride information
- **WebGPU stride support**: Kernels assume contiguous layout
- **View mutation**: No `strided_scatter_*` lowering
- **Slicing**: Not implemented

---

## Phase 1: ViewMeta Type Definition

### 1.1 Add ViewMeta to backend types

**File**: `src/backend/types.ts`

```typescript
/**
 * Metadata describing how a view maps to its base storage.
 * Per spec §2.3.
 */
export type ViewMeta = {
  /** The base tensor this view refers to */
  baseId: number;

  /** Byte offset from start of base storage */
  offsetBytes: number;

  /** Shape of the view */
  shape: number[];

  /** Bytes to skip for each dimension (row-major order) */
  stridesBytes: number[];

  /** True if memory layout is contiguous (enables fast paths) */
  isContiguous: boolean;
};

/**
 * Compute strides in elements (not bytes) for a contiguous tensor.
 */
export function computeContiguousStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Check if strides represent a contiguous layout for the given shape.
 */
export function checkContiguous(shape: number[], strides: number[]): boolean {
  const expected = computeContiguousStrides(shape);
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] <= 1) continue; // Size-1 dims don't matter
    if (strides[i] !== expected[i]) return false;
  }
  return true;
}
```

### 1.2 Add stride fields to BackendTensor

**File**: `src/backend/types.ts`

```typescript
export type BackendTensor = {
  shape: Shape;
  dtype: DType;

  // New fields for strided support
  strides?: number[];      // Element strides (not bytes)
  offset?: number;         // Element offset into storage
  isContiguous?: boolean;  // Cached contiguity flag
};
```

---

## Phase 2: IR Integration

### 2.1 Update IRNode to track ViewMeta

**File**: `src/engine/ir.ts`

```typescript
export type IRNode = {
  id: number;
  op: string;
  inputs: number[];
  shape?: number[];
  dtype?: DType;
  epoch: number;
  kind: "lazy_op" | "materialized" | "external";

  // New: View metadata for view-creating ops
  viewMeta?: {
    sourceNodeId: number;  // The node this is a view of
    offset: number;        // Element offset
    strides: number[];     // Element strides
  };
};
```

### 2.2 Update LazyIRNode similarly

**File**: `src/engine/lazy.ts`

Add `viewMeta` field to `LazyIRNode` with same structure.

### 2.3 View-creating ops must populate viewMeta

**File**: `src/runtime/engine.ts`

Update these ops to compute and attach viewMeta:
- `view(a, shape)` - Requires contiguous input, computes new strides
- `reshape(a, shape)` - Same as view
- `transpose(a, dims)` - Permutes strides according to dims
- `expand(a, shape)` - Sets stride=0 for broadcast dims
- `slice(a, starts, ends, steps)` - Computes offset and new strides (NEW OP)
- `squeeze(a, dim)` - Removes size-1 dims from shape/strides
- `unsqueeze(a, dim)` - Inserts size-1 dim with stride=0

---

## Phase 3: CPU Backend Updates

### 3.1 Tensor class updates

**File**: `src/backend/cpu/numeric.ts`

The CPU `Tensor` class already has `strides` and `offset`. Ensure all view operations properly track these.

### 3.2 Verify strided access in all ops

Most ops already use `readAtLinear()`/`writeAtLinear()` which handle strides. Verify:
- [ ] `add`, `sub`, `mul`, `div` - Already strided
- [ ] `matmul` - Already strided
- [ ] `sum`, `mean` - Check stride handling in reduction
- [ ] `gather`, `scatterAdd` - Already strided
- [ ] `relu`, `gelu`, `sigmoid`, etc. - Check unary ops

### 3.3 Add `contiguous()` operation

**File**: `src/backend/cpu/numeric.ts`

```typescript
contiguous(): Tensor {
  if (this.isContiguous()) {
    return this; // Already contiguous, return self
  }

  // Materialize to new contiguous storage
  const size = sizeOf(this.shape);
  const newData = new Float32Array(size);
  const newStrides = computeContiguousStrides(this.shape);

  for (let i = 0; i < size; i++) {
    newData[i] = this.readAtLinear(i);
  }

  return new Tensor(this.shape, newData, newStrides, 0);
}
```

### 3.4 Update `view()` to work with non-contiguous

Current `view()` throws for non-contiguous. Options:
1. Keep throwing (require explicit `contiguous()` call first)
2. Auto-call `contiguous()` internally
3. Support non-contiguous reshapes when mathematically valid

Recommendation: Option 1 (explicit) for predictability.

---

## Phase 4: WebGPU Backend Updates

### 4.1 Add stride uniforms to kernels

**File**: `src/backend/webgpu/index.ts`

Currently kernels use `gid.x` directly as linear index. Need to add:

```wgsl
struct TensorMeta {
  shape: array<u32, 4>,
  strides: array<u32, 4>,
  offset: u32,
  rank: u32,
}

fn strided_index(linear: u32, meta: TensorMeta) -> u32 {
  var idx = linear;
  var result = meta.offset;
  for (var d = meta.rank - 1; d >= 0; d--) {
    let coord = idx % meta.shape[d];
    idx = idx / meta.shape[d];
    result += coord * meta.strides[d];
  }
  return result;
}
```

### 4.2 Update elementwise kernels

All elementwise ops need to:
1. Accept stride metadata as uniforms
2. Use `strided_index()` for input reads
3. Output is always contiguous (no strided writes for elementwise)

### 4.3 Update matmul kernel

The tiled matmul already handles batched broadcasting. Extend to handle strided inputs:
- Add stride uniforms for A and B
- Modify tile loading to use strided access
- Output remains contiguous

### 4.4 Update fusion codegen

**File**: `src/backend/webgpu/fusion-codegen.ts`

Fused kernels need stride-aware loads:
```wgsl
// Instead of: let v0 = in0[idx];
// Use: let v0 = in0[strided_index(idx, meta0)];
```

### 4.5 Add `contiguous()` WebGPU implementation

Simple copy kernel that reads strided, writes contiguous.

---

## Phase 5: Slicing Support

### 5.1 Add slice operation

**File**: `src/runtime/engine.ts`

```typescript
slice(
  a: LazyRef,
  starts: number[],  // Start indices per dim
  ends: number[],    // End indices per dim (exclusive)
  steps?: number[],  // Step sizes (default 1)
): LazyRef {
  // Compute new shape
  const shape = starts.map((s, i) => Math.ceil((ends[i] - s) / (steps?.[i] ?? 1)));

  // Compute offset (in elements)
  const aStrides = getStrides(a);
  let offset = getOffset(a);
  for (let i = 0; i < starts.length; i++) {
    offset += starts[i] * aStrides[i];
  }

  // Compute new strides (multiply by step)
  const newStrides = aStrides.map((s, i) => s * (steps?.[i] ?? 1));

  return createViewNode(a, shape, newStrides, offset);
}
```

### 5.2 Backend implementations

CPU: Returns view with adjusted offset/strides (no copy)
WebGPU: Same - just metadata change

### 5.3 Gradient for slice

```typescript
// Backward: scatter gradient into zeros at sliced positions
sliceBackward(grad, inputShape, starts, ends, steps) {
  const zeros = zerosLike(inputShape);
  return scatterSlice(zeros, grad, starts, ends, steps);
}
```

---

## Phase 6: View Mutation Lowering (§4.4)

### 6.1 The Problem

When mutating a view in-place:
```typescript
const a = tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
const b = a.transpose([1, 0]);  // b is a view of a
b.add_(1);  // In-place add to transposed view
// a should now reflect the changes!
```

The mutation must write back to the correct positions in the base storage.

### 6.2 Lowering Strategy (per spec §4.4)

```
In-place op on view v with base b:
1. baseVal = base_load(b.baseId)
2. baseCopy = copy(baseVal)  // For version tracking
3. baseNew = strided_scatter_<op>(baseCopy, v.viewMeta, operands...)
4. base_store(b.baseId, baseNew)
```

### 6.3 Implement strided_scatter_add

**File**: `src/backend/cpu/numeric.ts`

```typescript
stridedScatterAdd(
  base: Tensor,           // The base storage
  viewMeta: ViewMeta,     // How the view maps to base
  values: Tensor,         // Values to add (same shape as view)
): Tensor {
  const result = base.clone();
  const viewShape = viewMeta.shape;
  const viewStrides = viewMeta.stridesBytes.map(b => b / 4); // Bytes to elements

  for (let i = 0; i < sizeOf(viewShape); i++) {
    // Convert linear index to view coordinates
    const coords = linearToCoords(i, viewShape);
    // Convert view coords to base offset using view strides
    let baseOffset = viewMeta.offsetBytes / 4;
    for (let d = 0; d < coords.length; d++) {
      baseOffset += coords[d] * viewStrides[d];
    }
    result.data[baseOffset] += values.readAtLinear(i);
  }

  return result;
}
```

### 6.4 Similar for other in-place ops

- `strided_scatter_sub`
- `strided_scatter_mul`
- `strided_scatter_div`
- `strided_scatter_copy` (for `copy_()`)

### 6.5 IR lowering pass

Add a pass in plan execution that detects in-place ops on views and lowers them to the strided_scatter pattern.

---

## Phase 7: Testing

### 7.1 Unit tests for ViewMeta

```typescript
describe("ViewMeta", () => {
  it("computes contiguous strides correctly");
  it("detects non-contiguous layouts");
  it("handles size-1 dimensions");
});
```

### 7.2 View operation tests

```typescript
describe("Strided views", () => {
  it("transpose creates correct strides");
  it("slice creates correct offset and strides");
  it("expand sets stride=0 for broadcast dims");
  it("chained views compose correctly");
  it("contiguous() materializes non-contiguous view");
});
```

### 7.3 View mutation tests

```typescript
describe("View mutation (§4.4)", () => {
  it("mutating transpose view updates base");
  it("mutating slice view updates base");
  it("mutating expanded view broadcasts correctly");
  it("version tracking works through view mutations");
});
```

### 7.4 WebGPU strided tests

```typescript
describe("WebGPU strided access", () => {
  it("elementwise ops work on transposed tensors");
  it("matmul works with strided inputs");
  it("fused kernels handle strided inputs");
});
```

---

## Implementation Order

1. **Phase 1**: ViewMeta type (1-2 hours)
2. **Phase 2**: IR integration (2-3 hours)
3. **Phase 3**: CPU backend updates (2-3 hours)
4. **Phase 5**: Slicing support (2-3 hours)
5. **Phase 4**: WebGPU backend updates (4-6 hours) - most complex
6. **Phase 6**: View mutation lowering (4-6 hours) - most complex
7. **Phase 7**: Testing throughout

**Total estimate**: 2-3 days of focused work

---

## Dependencies and Risks

### Dependencies
- Phase 2 depends on Phase 1
- Phase 3-6 depend on Phase 2
- Phase 4 (WebGPU) can be done in parallel with Phase 5-6

### Risks
1. **WebGPU performance**: Strided access may be slower than contiguous
   - Mitigation: Add `contiguous()` hints, auto-materialize before compute-heavy ops

2. **View mutation complexity**: Getting the lowering right is tricky
   - Mitigation: Start with CPU-only, extensive testing

3. **Fusion interaction**: Fused kernels with strided inputs add complexity
   - Mitigation: Initially disable fusion for non-contiguous inputs

---

## Future Enhancements

After basic strided support:
1. **Negative strides**: For `flip()` operation
2. **Advanced slicing**: NumPy-style `a[::2, 1::3]`
3. **As-strided**: Direct stride manipulation (dangerous but powerful)
4. **Memory format**: NHWC vs NCHW layout tracking
