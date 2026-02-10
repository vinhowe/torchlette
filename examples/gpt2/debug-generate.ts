/**
 * Test text generation with the fixed model.
 */
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

// Simple BPE tokenizer (just for testing - uses GPT-2 vocab)
import * as fs from "node:fs";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();

  // Load tokenizer vocab for decoding
  const vocabPath = path.join(modelDir, "vocab.json");
  const vocab: Record<string, number> = JSON.parse(fs.readFileSync(vocabPath, "utf-8"));
  const idToToken: Record<number, string> = {};
  for (const [token, id] of Object.entries(vocab)) {
    idToToken[id] = token;
  }

  // Simple greedy generation
  async function generate(prompt: number[], maxTokens: number): Promise<number[]> {
    const tokens = [...prompt];
    for (let i = 0; i < maxTokens; i++) {
      const input = api.tensorFromArray(tokens, [1, tokens.length], { device: "webgpu" });
      const { logits } = model.forwardWithLoss(input);
      const logitsData = Array.from(await logits.cpu());
      // Get logits for last position
      const vocabSize = 50257;
      const lastPos = tokens.length - 1;
      const lastLogits = logitsData.slice(lastPos * vocabSize, (lastPos + 1) * vocabSize);
      // Argmax
      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let j = 0; j < lastLogits.length; j++) {
        if (lastLogits[j] > maxVal) {
          maxVal = lastLogits[j];
          maxIdx = j;
        }
      }
      tokens.push(maxIdx);
      // Decode token for display
      const tokenStr = idToToken[maxIdx] ?? `[${maxIdx}]`;
      process.stdout.write(tokenStr.replace(/Ġ/g, " ").replace(/Ċ/g, "\n"));
    }
    return tokens;
  }

  // Test prompts
  // "Hello" = 15496
  console.log("=== Greedy generation from 'Hello' ===");
  process.stdout.write("Hello");
  await generate([15496], 30);
  console.log("\n");

  // "The" = 464
  console.log("=== Greedy generation from 'The' ===");
  process.stdout.write("The");
  await generate([464], 30);
  console.log("\n");

  // "To be or not to be" = [2514, 307, 393, 407, 284, 307]
  console.log("=== Greedy generation from 'To be or not to be' ===");
  process.stdout.write("To be or not to be");
  await generate([2514, 307, 393, 407, 284, 307], 30);
  console.log("\n");

  console.log("=== Done ===");
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
