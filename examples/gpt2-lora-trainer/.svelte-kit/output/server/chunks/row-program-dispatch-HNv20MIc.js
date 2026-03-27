import { r as requireContext, a as dtypeBytes, b as resolveOutputBuffer, c as createTileKernelDispatcher, e as dtypeToTileIR, p as perRowKernel, f as init_buffer_arena, g as init_shape_utils, h as init_webgpu_state, j as init_fusion_tile_ir, k as init_tile_ir, l as applyFusedOp, m as init_tile_dispatch, o as onTeardown } from "./tensor-DNI3XZeL.js";
init_buffer_arena();
init_shape_utils();
init_webgpu_state();
function isRPValue(e) {
  return "kind" in e;
}
init_fusion_tile_ir();
init_tile_ir();
const WG = 256;
function rowProgramToSpec(program) {
  const bindings = {};
  for (let i = 0; i < program.inputs.length; i++) bindings[`in${i}`] = {
    storage: "read",
    type: dtypeToTileIR(program.inputs[i].dtype)
  };
  bindings["output"] = {
    storage: "read_write",
    type: dtypeToTileIR(program.output.dtype)
  };
  const needsF16 = program.inputs.some((inp) => inp.dtype === "f16") || program.output.dtype === "f16";
  return perRowKernel({
    name: `rowProg_${program.cacheKey}`,
    bindings,
    enableF16: needsF16 || void 0,
    kernel(ctx, _row, tid, D, base) {
      const reduceResults = [];
      for (const phase of program.phases) if (phase.kind === "reduce") {
        const result = ctx.wgReduce(phase.reduceOp, tid, D, WG, (i) => emitExpr(ctx, phase.bodyExpr, base.add(i), reduceResults));
        let finalResult = result;
        if (phase.isMean) finalResult = result.div(D.toF32());
        const named = ctx.emitLet(`r${reduceResults.length}`, finalResult);
        reduceResults.push(named);
      } else if (phase.scalarOutput) ctx.ifThen(tid.eq(ctx.u32(0)), () => {
        const val = emitExpr(ctx, phase.bodyExpr, base, reduceResults);
        ctx.emitStore("output", _row, val);
      });
      else ctx.stridedFor(tid, D, WG, (i) => {
        const val = emitExpr(ctx, phase.bodyExpr, base.add(i), reduceResults);
        ctx.emitStore("output", base.add(i), val);
      });
    }
  });
}
function emitExpr(ctx, expr, elementOffset, reduceResults) {
  if (isRPValue(expr)) switch (expr.kind) {
    case "input":
      return ctx.load(`in${expr.bufferIndex}`, elementOffset);
    case "reduceResult":
      return reduceResults[expr.phaseIndex];
    case "const":
      return ctx.f32(expr.value);
  }
  const inputExprs = expr.inputs.map((inp) => emitExpr(ctx, inp, elementOffset, reduceResults));
  return applyFusedOp(ctx, expr.op, inputExprs);
}
init_shape_utils();
init_tile_dispatch();
const kernelCache = /* @__PURE__ */ new Map();
function getOrCreateKernel(program) {
  let kernel = kernelCache.get(program.cacheKey);
  if (!kernel) {
    kernel = createTileKernelDispatcher(rowProgramToSpec(program));
    kernelCache.set(program.cacheKey, kernel);
  }
  return kernel;
}
function dispatchRowProgram(program, inputBuffers, numRows, dimSize) {
  const ctx = requireContext();
  const lastPhase = program.phases[program.phases.length - 1];
  const outBytes = (lastPhase.kind === "write" && lastPhase.scalarOutput ? numRows : numRows * dimSize) * dtypeBytes(program.output.dtype);
  const outBuffer = resolveOutputBuffer(ctx.device, outBytes, inputBuffers);
  const buffers = {};
  for (let i = 0; i < inputBuffers.length; i++) buffers[`in${i}`] = inputBuffers[i];
  buffers["output"] = outBuffer;
  getOrCreateKernel(program).dispatch(buffers, {
    num_rows: numRows,
    feature_dim: dimSize
  });
  return outBuffer;
}
function resetRowProgramKernelState() {
  for (const k of kernelCache.values()) k.reset();
  kernelCache.clear();
}
onTeardown(resetRowProgramKernelState);
export {
  dispatchRowProgram
};
