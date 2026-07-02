/**
 * Qwen3 chat inference server (demo).
 *
 * Node/Dawn: loads Qwen3-1.7B f32 once, serves POST /api/chat as an SSE token
 * stream. Generation (prefill + KV-cache decode + sampling + chat template)
 * comes from the shared qwen3-browser package — the same code the in-browser
 * mode runs. Single-flight: requests queue behind one generation at a time.
 *
 * Run: npx tsx examples/qwen3-chat/server.ts   (port 8787)
 */

import * as http from "node:http";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { AutoTokenizer, env as hfEnv } from "@huggingface/transformers";
import { generateChat, type ChatMessage } from "qwen3-browser";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "../qwen3/loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PORT = 8787;

async function main() {
  console.log("Loading tokenizer…");
  hfEnv.localModelPath = path.join(__dirname, "../../ckpts");
  hfEnv.allowRemoteModels = false;
  const tokenizer = await AutoTokenizer.from_pretrained("qwen3-1.7b");
  const tokenizerLike = {
    encode: (text: string) => tokenizer.encode(text) as number[],
    decode: (ids: number[], options?: { skip_special_tokens?: boolean }) =>
      tokenizer.decode(ids, options) as string,
  };

  console.log("Initializing WebGPU…");
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  console.log("Loading Qwen3-1.7B (f32)…");
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 2048 });

  let queue: Promise<void> = Promise.resolve();

  const server = http.createServer((req, res) => {
    if (req.method === "GET" && req.url === "/api/health") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true, model: "Qwen3-1.7B", backend: "torchlette/webgpu (Node Dawn)" }));
      return;
    }
    if (req.method !== "POST" || req.url !== "/api/chat") {
      res.writeHead(404).end();
      return;
    }

    let body = "";
    req.on("data", (c) => {
      body += c;
    });
    req.on("end", () => {
      let messages: ChatMessage[];
      try {
        messages = JSON.parse(body).messages;
        if (!Array.isArray(messages)) throw new Error("messages must be an array");
      } catch (e) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: String(e) }));
        return;
      }

      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      let aborted = false;
      // NB: req 'close' fires when the BODY stream ends (immediately) on modern
      // Node — client disconnect is res 'close' with the response unfinished.
      res.on("close", () => {
        if (!res.writableEnded) aborted = true;
      });
      const send = (event: Record<string, unknown>) => {
        if (!aborted && !res.writableEnded) res.write(`data: ${JSON.stringify(event)}\n\n`);
      };

      queue = queue
        .then(async () => {
          const stats = await generateChat(
            api,
            model,
            tokenizerLike,
            messages,
            {
              onDelta: (delta) => send({ delta }),
              onReplace: (text) => send({ replace: text }),
            },
            { isAborted: () => aborted },
          );
          send({ done: true, stats });
        })
        .catch((e) => {
          console.error("generation error:", e);
          send({ error: String(e) });
        })
        .then(() => {
          if (!res.writableEnded) res.end();
        });
    });
  });

  server.listen(PORT, "127.0.0.1", () => {
    console.log(`qwen3-chat inference server on http://127.0.0.1:${PORT}`);
  });
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
