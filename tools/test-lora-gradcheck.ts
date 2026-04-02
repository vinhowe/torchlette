/**
 * Gradcheck on LoRA components to find which backward is wrong.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { gradcheck } from "../src/testing/gradcheck";

async function check(
  api: Torchlette,
  name: string,
  fn: (...args: any[]) => any,
  inputs: any[],
) {
  const result = await gradcheck(api, fn, inputs, {
    eps: 1e-3,
    atol: 1e-2,
    rtol: 1e-2,
  });
  console.log(`[${result.pass ? "PASS" : "FAIL"}] ${name}`);
  for (const r of result.inputs) {
    const status = r.maxAbsErr < 0.01 ? "ok" : "BAD";
    console.log(
      `  ${status} shape=${JSON.stringify(r.shape)} absErr=${r.maxAbsErr.toExponential(2)} analytical=${r.analyticalAtWorst.toExponential(2)} numerical=${r.numericalAtWorst.toExponential(2)}`,
    );
  }
  return result.pass;
}

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false });

  // 1. Simple matmul
  await check(
    api,
    "matmul x @ W^T",
    (x, W) => api.sum(api.matmul(x, W.transpose({ dim0: 0, dim1: 1 }))),
    [
      api.randn([1, 4, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([16, 8], { device: "webgpu", requiresGrad: true }),
    ],
  );

  // 2. Chained matmul (LoRA core: x @ A^T @ B^T)
  await check(
    api,
    "x @ A^T @ B^T",
    (x, A, B) => {
      const h = api.matmul(x, A.transpose({ dim0: 0, dim1: 1 }));
      return api.sum(api.matmul(h, B.transpose({ dim0: 0, dim1: 1 })));
    },
    [
      api.randn([1, 4, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([3, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([16, 3], { device: "webgpu", requiresGrad: true }),
    ],
  );

  // 3. detach + add
  await check(
    api,
    "detach(base) + lora",
    (base, lora) => api.sum(api.add(base.detach(), lora)),
    [
      api.randn([1, 4, 16], { device: "webgpu", requiresGrad: true }),
      api.randn([1, 4, 16], { device: "webgpu", requiresGrad: true }),
    ],
  );

  // 4. Full LoRA with detach (B=0 like real init)
  await check(
    api,
    "full LoRA (B=0)",
    (x, W, bias, A, B) => {
      const base = api.add(
        api.matmul(x, W.transpose({ dim0: 0, dim1: 1 })),
        bias,
      );
      const lora = api.matmul(
        api.matmul(x, A.transpose({ dim0: 0, dim1: 1 })),
        B.transpose({ dim0: 0, dim1: 1 }),
      );
      return api.sum(api.add(base.detach(), lora));
    },
    [
      api.randn([1, 4, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([16, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([16], { device: "webgpu", requiresGrad: true }),
      api.randn([3, 8], { device: "webgpu", requiresGrad: true }),
      api.zeros([16, 3], { device: "webgpu", requiresGrad: true }),
    ],
  );

  // 5. Full LoRA with scaling
  await check(
    api,
    "full LoRA with scaling",
    (x, A, B) => {
      const lora = api.matmul(
        api.matmul(x, A.transpose({ dim0: 0, dim1: 1 })),
        B.transpose({ dim0: 0, dim1: 1 }),
      );
      const scale = api.tensorFromArray([0.5], []);
      return api.sum(api.mul(lora, scale));
    },
    [
      api.randn([1, 4, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([3, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([16, 3], { device: "webgpu", requiresGrad: true }),
    ],
  );

  // 6. add + matmul chain (base + lora through more ops)
  await check(
    api,
    "layernorm-like + matmul",
    (x, W) => {
      const mean = x.mean(-1, true);
      const centered = api.sub(x, mean);
      const out = api.matmul(centered, W.transpose({ dim0: 0, dim1: 1 }));
      return api.sum(out);
    },
    [
      api.randn([1, 4, 8], { device: "webgpu", requiresGrad: true }),
      api.randn([16, 8], { device: "webgpu", requiresGrad: true }),
    ],
  );

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
