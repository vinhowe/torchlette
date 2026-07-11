# Appendix A — paper Triton capability profile

**Scope.** Ordinary public `triton` / `triton.language` on the public `main`
documentation inspected 2026-07-11; experimental Gluon is noted but is not silently
counted as the v2 surface. **Determination** means the emitted program fixes the stated
property; **request** means Triton receives a meta-parameter or compiler attribute but
TTGIR chooses the physical realization; **refused** means ordinary `tl.*` has no surface
for that property. A split verdict is itself a §2 representation bug. Public `main` is
not a stable version contract; a v2 profile must pin a Triton release/commit and target
architecture. Anything described here as limited or experimental must be rechecked then.

Primary surface facts: `triton.Config` exposes `num_warps`, `num_stages`, `num_ctas`, and
`maxnreg`; `tl.range` exposes loop-local staging, unrolling, flattening, LICM control, and
limited automatic warp specialization; `tl.load` exposes cache/eviction/volatile hints;
`tl.debug_barrier` is block-wide; and atomics expose memory semantics and scope.
([Config](https://triton-lang.org/main/python-api/generated/triton.Config.html),
[range](https://triton-lang.org/main/python-api/generated/triton.language.range.html),
[load](https://triton-lang.org/main/python-api/generated/triton.language.load.html),
[barrier](https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html),
[atomic add](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html))

## Moves

| v1 move | Verdict | What Triton actually controls |
|---|---|---|
| `tile` | **determination** for logical block shape; **request** for its physical layout — **A-R1** | `tl.arange`, masks, pointer arithmetic, and `tl.constexpr` fix the logical block. Lane/warp/register ownership and shared swizzles are TTGIR encodings chosen downstream. “Tile” conflates those two facts. |
| `stream` | **determination** for eliminating an intermediate dispatch/store and emitting a source loop; **request** for physical residency — **A-R2** | The realizer can emit repeated loads in a loop, but LICM, caching, staging, spills, and load scheduling remain compiler decisions. “Stream” needs a semantic no-materialization contract separate from a residency request. |
| `recolor` | **refused** if, as P2 implies, it pins an accumulator/operand to registers or remaps lanes; **determination** if it merely rewrites logical indices — **A-R3** | Stable `tl.*` does not accept user tensor-layout or register-placement attributes. The move has no schema, so it has no single verdict. |
| `fuse` | **determination** within its legality horizon | Emitting one JIT kernel fixes one launch and removes global intermediates. It cannot fuse a dependence that requires synchronization between program instances: Triton has no grid-wide barrier inside a kernel. Island membership must be changed above this realizer, not again here. |
| `pack` | **determination** for horizontal multi-tensor work in one emitted program; **refused** for a demanded SIMD/`vec4` packing — **A-R4** | Pointer/index code can pack independent work. Physical vector load width is compiler-selected; `tl.*` has no `vec4` knob. The unqualified move name conflates the two. |
| `role-partition` | **request**, narrowly; otherwise **refused** — **A-R5** | `tl.range(..., warp_specialize=True)` asks the compiler to partition a simple matmul loop and is currently documented only for Blackwell. It does not describe named producer/consumer roles. Experimental Gluon has explicit warp-specialized regions, but that is a different, unstable surface. |
| `pipeline` | **request** | Kernel `num_stages` requests dot-load pipelining; loop-local `tl.range(num_stages=...)` requests broader load pipelining. TTGIR assigns latencies, schedules loops, inserts/lowers fences, and may realize a different instruction schedule. |

## Decorations

| v1 decoration | Verdict | Triton mapping / truncation |
|---|---|---|
| tile sizes | **determination** only as logical tensor extents — **A-R1** | `BLOCK_*: tl.constexpr` and `tl.arange` fix logical extents. They do not fix per-thread tiles, layout, or instruction shape. |
| vec width | **refused** — **A-R6** | There is no public exact vector-width parameter. Alignment/contiguity assertions and pointer shape inform coalescing; TTGIR chooses load width. The repo's WGSL `array<vec4<T>>` reinterpretation is not portable. |
| workgroup dimensions | **refused** as WGSL `[x,y,z]`; nearest mapping is **request** — **A-R7** | `num_warps` supplies a CTA warp budget, not an x/y/z local-ID geometry or a user mapping of elements to those threads. |
| operand residency | **refused** — **A-R8** | Global pointer accesses are explicit, but register versus shared staging of block values is encoded/allocated in TTGIR. There is no stable `tl.*` “put operand B in shared” surface. |
| pipeline depth | **request** — **A-R9** | `num_stages` is kernel- or loop-scoped and the two forms have different meanings. One scalar on the whole ScheduleState cannot represent multiple pipelined loops or “no pipeline.” |
| unroll (named in §2's `DecorationVector`) | **request** | `tl.range(loop_unroll_factor=n)` and `tl.static_range` guide IR unrolling. This is loop-local, not a kernel-global Boolean/scalar. |

## Enumerated axes, atoms, and lemmas

| v1 element | Verdict | Triton mapping / truncation |
|---|---|---|
| memory `global` | **determination** | Pointer loads/stores are explicit global-memory effects. |
| memory `workgroup-shared` | **refused** as an explicit placement — **A-R8** | Shared allocation, layout, and staging are TTGIR/backend decisions. Tensor descriptors may select TMA, but do not expose a portable shared variable. |
| reserved `register-explicit`, `distributed-shared/cluster` | **refused** | `maxnreg` is a ceiling, not placement. `num_ctas` requests a cluster size on SM90+, not a public distributed-shared allocation contract. |
| sync `workgroup` | **determination** for an explicit block barrier | `tl.debug_barrier()` is block-wide. The axis still omits primitive, memory order, and whether it describes a barrier or an atomic scope — **A-R10**. |
| reserved sync `subgroup`, `cluster`, `grid` | **refused** in ordinary `tl.*` | Atomics instead offer `cta`/`gpu`/`sys` visibility scopes; those do not mean subgroup/cluster/grid barriers. Grid synchronization requires another launch/cooperative mechanism. |
| hierarchy `workgroup` | **determination** at program/CTA granularity | `tl.program_id(axis)` fixes the program instance and launch grid. |
| hierarchy `invocation`, `subgroup` | **refused** — **A-R11** | Ordinary Triton intentionally hides lane/thread IDs and assigns block elements through internal layouts. |
| reserved hierarchy `warp-role` | **request** | Limited automatic warp specialization is a compiler request; named roles require experimental Gluon or lower IR. |
| reserved hierarchy `cluster` | **request** | `num_ctas` requests blocks per cluster on supported hardware; it does not determine the cluster layout or shared-memory protocol. |
| `atomicAddF32-CAS` atom | **determination** for either native `tl.atomic_add` or an explicitly emitted `tl.atomic_cas` retry loop; **A-R12** | The semantic operation is portable, but the atom's name bakes the WGSL realization into the object. Triton normally has native float atomic add and explicit `sem`/`scope`. |
| “subgroup ops (feature-gated)” atom family | **refused** as stated — **A-R13** | Block reductions may lower to warp instructions, but stable `tl.*` has no generic subgroup value, lane ID, or exact shuffle contract. A family name without operation/signature/scope cannot be profiled. |
| admitted-lemma mechanism | **determination** at the host rewrite layer | A lemma is applied before emission; Triton compiles the transformed algorithm. Capability is per lemma's resulting operations, not a property of “admission,” and proof/differential status is outside the realizer. |

## Authority horizon

The horizon is after semantic TTIR and before/inside TTGIR. Source code determines
arithmetic, pointer/index expressions, masks, explicit control flow, launch grid,
dispatch count, explicit atomics, and the block barrier. `num_warps`, `num_stages`,
`num_ctas`, `maxnreg`, cache modifiers, and loop attributes are requests or bounds.
TTGIR/backend passes independently choose tensor encodings, element-to-register/lane/warp
layouts, coalescing and vector width, shared-memory allocation/swizzles, tensor-core
lowering, thread locality, loop fusion/LICM, software-pipeline instruction order,
fence insertion, register allocation, and spills. Triton's own NVIDIA backend visibly
runs coalescing, thread-locality, matmul acceleration, layout-conversion removal,
latency assignment, loop scheduling, pipelining, fence insertion, and shared-memory
allocation after TTIR→TTGIR conversion
([backend pipeline](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py)).
`Config.ir_override` can bypass this horizon with TTGIR/LLVM/PTX, but counting that escape
hatch would abandon §4's “Triton pays the CUDA tax” premise and create a second low-level
realizer.

## Representation bugs fed to the review memo

- **A-R1–A-R5:** five move names mix semantic rewrites with physical placement or leave
  the target undefined; a capability profile cannot assign one verdict.
- **A-R6–A-R9:** WGSL `vec4`, x/y workgroup geometry, explicit residency, and one global
  pipeline-depth scalar do not map to Triton's surface.
- **A-R10–A-R13:** scope lacks primitive/order, hierarchy assumes exposed invocations,
  and both atom declarations are backend- or family-shaped rather than semantic.
- **A-R14 (missing surface):** §2 has no coordinates for `num_warps`, `num_ctas`,
  `maxnreg`, cache/eviction policy, atomic semantics/scope, or loop-local pipeline,
  unroll, flatten, and LICM controls. These cannot be recovered from `workgroup dims` or
  one `pipeline depth` without deformation.
- **A-R15 (published counterexample):** §2 has no program-grid traversal/remapping
  coordinate. Triton's published matmul remaps program IDs in row groups to improve L2
  reuse, reporting over 10% on A100, and `tl.swizzle2d` exposes the same class of mapping
  ([tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html),
  [API](https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html)).
  This is logical index remapping—fully expressible with WGSL `workgroup_id` arithmetic—
  not accumulator `recolor`, tiling, streaming, fusion, packing, role partitioning, or
  pipelining. It is review finding **R4**.
