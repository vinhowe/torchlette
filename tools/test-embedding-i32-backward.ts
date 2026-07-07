/**
 * Test: does embedding() + scatterAdd backward correctly route gradients
 * back to the right embedding rows when indices are i32?
 */
import { Torchlette, initWebGPU } from "../src/index";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  // Weight: 5 rows × 3 cols, initialized to distinct values per row.
  const weight = api.tensorFromArray(
    [
      0, 0, 0,   // row 0
      1, 1, 1,   // row 1
      2, 2, 2,   // row 2
      3, 3, 3,   // row 3
      4, 4, 4,   // row 4
    ],
    [5, 3],
    { requiresGrad: true },
  );

  // Lookup indices [0, 2, 4, 1] as i32 tensor.
  const tokensI32 = api.tensorFromArray([0, 2, 4, 1], [4], { dtype: "i32" });
  // Same as f32 tensor.
  const tokensF32 = api.tensorFromArray([0, 2, 4, 1], [4]);

  // Forward with i32
  await api.beginStep();
  const emb_i32 = api.embedding(weight, tokensI32);
  const loss_i32 = emb_i32.sum();
  await loss_i32.backward();
  const grad_i32 = await weight.grad!.cpu();
  await api.endStep();

  // Reset grad
  weight.grad?.fill_(0);

  // Forward with f32
  await api.beginStep();
  const emb_f32 = api.embedding(weight, tokensF32);
  const loss_f32 = emb_f32.sum();
  await loss_f32.backward();
  const grad_f32 = await weight.grad!.cpu();
  await api.endStep();

  console.log("grad w/ i32 tokens:", Array.from(grad_i32));
  console.log("grad w/ f32 tokens:", Array.from(grad_f32));
  console.log("expected: row 0 = 1, row 1 = 1, row 2 = 1, row 3 = 0, row 4 = 1 (per col)");
  console.log("         [1,1,1, 1,1,1, 1,1,1, 0,0,0, 1,1,1]");

  // Compare embedding outputs too
  const embI32Data = await emb_i32.cpu();
  const embF32Data = await emb_f32.cpu();
  console.log("emb i32:", Array.from(embI32Data));
  console.log("emb f32:", Array.from(embF32Data));
  console.log("expected: [0,0,0, 2,2,2, 4,4,4, 1,1,1]");

  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
