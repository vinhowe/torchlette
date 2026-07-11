/**
 * Dump a genuine, CPU-built GPT-2 forward plan as schedule-editor ground truth.
 *
 * We collect a cold forward's real lazy plan (including deterministic parameter
 * initialization), run the production fusion segmenter, reify its partition,
 * and serialize stable final-plan positions. No GPU or model download is
 * required, and the captured plan is never executed.
 *
 *   TORCHLETTE_CPU_ONLY=1 pnpm exec tsx tools/t-schedule-dump.ts [output.json]
 *   pnpm exec tsx tools/t-schedule-dump.ts --hash-vector
 */
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import {
  computePlanFingerprint,
  type Island,
  type IslandKind,
  partitionBoundaryHash,
  reifyPartition,
  segmentPlanForExecution,
} from "../src/compiler/fusion-detect";
import { buildMergedPlan } from "../src/executor/plan-builder";
import { Torchlette } from "../src/frontend/torchlette";
import type { LazyIRNode } from "../src/graph/types";

const MODEL_CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 64,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0,
};
const BATCH = 1;
const SEQ = 32;

type SegmentDescriptor = { kind: IslandKind; finalPoss: number[] };

function hex(value: number): string {
  return `0x${value.toString(16).padStart(8, "0")}`;
}

function nodeForTensor(tensor: unknown): LazyIRNode {
  const ref = (
    tensor as { _unwrap(): { lazyRef: { kind: string; node?: LazyIRNode } } }
  )._unwrap().lazyRef;
  if (ref.kind !== "pending" || !ref.node) {
    throw new Error("Expected an unexecuted lazy tensor root");
  }
  return ref.node;
}

function descriptors(
  nodes: LazyIRNode[],
  segments: ReturnType<typeof segmentPlanForExecution>,
): SegmentDescriptor[] {
  const position = new Map(nodes.map((node, pos) => [node, pos]));
  return segments.map((segment) => {
    const members =
      segment.kind === "fused" ? segment.group.nodes : segment.nodes;
    return {
      kind: segment.kind,
      finalPoss: members
        .map((node) => {
          const pos = position.get(node);
          if (pos === undefined)
            throw new Error("Segment node absent from plan");
          return pos;
        })
        .sort((a, b) => a - b),
    };
  });
}

async function dump(outputPath: string): Promise<void> {
  const torch = new Torchlette("cpu", { enableFusion: true });
  const model = new GPT2(torch, MODEL_CONFIG, { device: "cpu" });
  model.eval();

  const tokens = Array.from(
    { length: BATCH * SEQ },
    (_, i) => (i * 37 + 11) % MODEL_CONFIG.vocabSize,
  );
  const input = torch.tensorFromArray(tokens, [BATCH, SEQ], {
    device: "cpu",
    dtype: "i32",
  });
  const logits = model.forward(input);
  const plan = buildMergedPlan([nodeForTensor(logits)]);
  const segments = segmentPlanForExecution(plan.nodes);
  const partition = reifyPartition(descriptors(plan.nodes, segments));
  const fingerprint = computePlanFingerprint(plan.nodes);

  const payload = {
    meta: {
      model: "gpt2-tiny-2l-64d",
      step: "cold-forward",
      planFingerprint: `${hex(fingerprint.primary)}:${hex(fingerprint.secondary)}`,
    },
    partition,
    nodes: plan.nodes.map((node, pos) => ({
      pos,
      op: node.op,
      shape: node.shape,
      dtype: node.dtype,
      ...(node.module ? { label: node.module } : {}),
    })),
  };

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  process.stderr.write(
    `wrote ${outputPath}: ${plan.nodes.length} nodes, ${partition.islands.length} islands, boundaryHash=${hex(partition.boundaryHash)}\n`,
  );
}

function printHashVector(): void {
  const islands: Island[] = [
    { kind: "sequential", members: [0, 3, 4] },
    { kind: "fused", members: [7, 8, 11, 12] },
    { kind: "reduction", members: [19] },
  ];
  process.stdout.write(
    `${JSON.stringify({ islands, boundaryHash: partitionBoundaryHash(islands) }, null, 2)}\n`,
  );
}

async function main(): Promise<void> {
  if (process.argv[2] === "--hash-vector") {
    printHashVector();
    return;
  }
  const output = path.resolve(
    process.argv[2] ??
      "examples/schedule-editor/public/data/gpt2-tiny-forward.json",
  );
  await dump(output);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
