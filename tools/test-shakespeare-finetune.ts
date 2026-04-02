/**
 * Shakespeare finetuning test: train on Shakespeare, generate Shakespeare-like text.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { generateTokens } from "../examples/gpt2-lora-trainer/src/lib/torchlette/inference";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, SGD } from "../src/optim";

const CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 6,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true });
  const d = path.join(process.cwd(), "models", "distilgpt2");
  const tok = new GPT2Tokenizer();
  tok.load(
    JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
    fs
      .readFileSync(path.join(d, "merges.txt"), "utf-8")
      .split("\n")
      .filter((l) => l && !l.startsWith("#")),
  );

  const model = new GPT2WithLoRA(
    api,
    CONFIG,
    { rank: 64, alpha: 64 },
    "webgpu",
  );
  const buf = fs.readFileSync(path.join(d, "model.safetensors"));
  const hl = Number(
    new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true),
  );
  const hdr = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hl)));
  const w = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [n, m] of Object.entries(hdr) as [string, any][]) {
    if (n === "__metadata__" || m.dtype !== "F32") continue;
    const r = buf.subarray(
      8 + hl + m.data_offsets[0],
      8 + hl + m.data_offsets[1],
    );
    w.set(n.replace(/^transformer\./, ""), {
      data: new Float32Array(new Uint8Array(r).slice().buffer),
      shape: m.shape,
    });
  }
  model.loadBaseWeights(w);
  await api.markStep();

  // Shakespeare training text — Coriolanus Act 1 Scene 1
  const shakespeare = `First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even to the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved
the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? the matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do,
which now we'll show 'em in deeds. They say poor
suitors have strong breaths: they shall know we
have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state, whose course will on
The way it takes, cracking ten thousand curbs
Of more strong link asunder than can ever
Appear in your impediment. For the dearth,
The gods, not the patricians, make it, and
Your knees to them, not arms, must help.`;

  const tokens = tok.encode(shakespeare);
  console.log(`Training on ${tokens.length} Shakespeare tokens\n`);

  // Generate BEFORE training
  console.log("=== Before Training ===");
  model.train(false);
  let gen = "";
  for await (const t of generateTokens(api, model, tok, "First Citizen:", {
    maxNewTokens: 50,
    temperature: 0.7,
    topK: 40,
  }))
    gen += t;
  console.log(`"First Citizen:${gen}"\n`);

  // Train
  model.train(true);
  const loraParams = model.getLoRAParameters();
  const optimizer = new SGD(loraParams, { lr: 0.05 }, api);
  const seqLen = Math.min(128, tokens.length - 1);

  console.log("=== Training 100 steps ===");
  let dataIdx = 0;
  for (let step = 0; step < 300; step++) {
    // Cycle through text sequentially
    if (dataIdx + seqLen + 1 > tokens.length) dataIdx = 0;
    await api.beginStep();
    const input = api.tensorFromArray(
      tokens.slice(dataIdx, dataIdx + seqLen),
      [1, seqLen],
      {
        device: "webgpu",
      },
    );
    const target = api.tensorFromArray(
      tokens.slice(dataIdx + 1, dataIdx + seqLen + 1),
      [1, seqLen],
      { device: "webgpu" },
    );
    dataIdx += seqLen;
    const { loss } = model.forwardWithLoss(input, target);
    const lossVal = await loss.item();
    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    if (step % 50 === 0 || step === 299) {
      console.log(`  Step ${step}: loss=${lossVal.toFixed(4)}`);
    }
  }

  // Generate AFTER training
  console.log("\n=== After Training ===");
  model.train(false);

  for (const prompt of [
    "First Citizen:",
    "All:",
    "Second Citizen:",
    "We are",
  ]) {
    let text = "";
    for await (const t of generateTokens(api, model, tok, prompt, {
      maxNewTokens: 60,
      temperature: 0.7,
      topK: 40,
    }))
      text += t;
    console.log(`"${prompt}${text}"\n`);
  }

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
