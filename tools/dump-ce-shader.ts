import { initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  // Force kernel compile
  const { compileTileKernel } = await import("../src/backend/webgpu/tile-compiler");
  const { perRowKernel, WORKGROUP_SIZE } = await import("../src/backend/webgpu/tile-ir") as any;
  const WG = 256;
  const spec = perRowKernel({
    name: "ceFwd",
    bindings: {
      logits: { storage: "read", type: "f32" },
      targets: { storage: "read", type: "i32" },
      loss: { storage: "read_write", type: "f32" },
    },
    rowUniform: "batch_size",
    dimUniform: "vocab_size",
    uniforms: { ignore_index: "i32" },
    kernel(ctx: any, row: any, tid: any, V: any, base: any) {
      const tI = ctx.emitLet("t_i", ctx.load("targets", row));
      const ignoreIdx = ctx.uniform("ignore_index", "i32");
      const isIgnored = ctx.emitLet("is_ignored", tI.eq(ignoreIdx));
      const tSafe = ctx.emitLet("t_safe", isIgnored.select(ctx.u32(0), tI.toU32()));
      const rowMax = ctx.emitLet(
        "row_max",
        ctx.wgReduce("max", tid, V, WG, (i: any) => ctx.load("logits", base.add(i))),
      );
      const logSumExp = ctx.emitLet(
        "log_sum_exp",
        ctx.wgReduce("sum", tid, V, WG, (i: any) =>
          ctx.load("logits", base.add(i)).sub(rowMax).exp(),
        ).log(),
      );
      const rawLoss = ctx.load("logits", base.add(tSafe)).sub(rowMax).sub(logSumExp).neg();
      ctx.guardedStore("loss", tid.eq(ctx.u32(0)), row, isIgnored.select(ctx.f32(0.0), rawLoss));
    },
  });
  const wgsl = compileTileKernel(spec);
  console.log(wgsl);
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
