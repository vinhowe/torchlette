import { Torchlette } from "../src/frontend/torchlette";

const api = new Torchlette("cpu");

async function main() {
  // Simple chain: matmul + softmax backward on CPU
  const x = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
    requiresGrad: true,
  });
  const w = api.tensorFromArray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [3, 2], {
    requiresGrad: true,
  });

  console.log("=== matmul + softmax ===");
  const y = api.matmul(x, w);
  const s = api.softmax(y, -1);
  const loss = api.sum(s);
  console.log("loss:", await loss.item());
  try {
    await loss.backward();
    console.log("backward OK");
    console.log("x.grad:", x.grad ? (await x.grad.cpu()).slice(0, 4) : "null");
    console.log("w.grad:", w.grad ? (await w.grad.cpu()).slice(0, 4) : "null");
  } catch (e: any) {
    console.error("FAIL:", e.message);
  }

  // Try linear
  console.log("\n=== linear ===");
  const x2 = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
    requiresGrad: true,
  });
  const w2 = api.tensorFromArray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3], {
    requiresGrad: true,
  });
  const b2 = api.tensorFromArray([0.01, 0.02], [2], { requiresGrad: true });
  try {
    const out = api.linear(x2, w2, b2);
    const loss2 = api.sum(out);
    console.log("loss:", await loss2.item());
    await loss2.backward();
    console.log("backward OK");
  } catch (e: any) {
    console.error("FAIL:", e.message);
  }

  // Try a chain that would trigger the reshape bug
  console.log("\n=== narrow + reshape ===");
  const x3 = api.tensorFromArray(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [3, 4],
    { requiresGrad: true },
  );
  try {
    const narrowed = api.narrow(x3, 1, 0, 2); // [3, 2]
    const reshaped = api.reshape(narrowed, [6]); // [6]
    const loss3 = api.sum(reshaped);
    console.log("loss:", await loss3.item());
    await loss3.backward();
    console.log("backward OK");
    console.log(
      "x3.grad:",
      x3.grad ? (await x3.grad.cpu()).slice(0, 6) : "null",
    );
  } catch (e: any) {
    console.error("FAIL:", e.message);
  }

  // Try mean backward
  console.log("\n=== mean backward ===");
  const x4 = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
    requiresGrad: true,
  });
  try {
    const m = api.mean(x4, { dim: 1, keepdim: true });
    const loss4 = api.sum(m as any);
    console.log("loss:", await loss4.item());
    await loss4.backward();
    console.log("backward OK");
    console.log("x4.grad:", x4.grad ? await x4.grad.cpu() : "null");
  } catch (e: any) {
    console.error("FAIL:", e.message);
  }

  process.exit(0);
}
main();
