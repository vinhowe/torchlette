import attentionFixture from "../../../public/data/ncd/attention-naive.term.json";
import { deriveFlashAttention } from "./fa-script";
import { applyMove, cloneTerm, napkinCost, onlineSoftmaxLemma } from "./model";
import type {
  LemmaRelabeling,
  NapkinCost,
  NcdBox,
  NcdTerm,
  PartitionDecoration,
} from "./types";

export type GameLevelId = "fuse-chain" | "layernorm" | "softmax" | "attention";

export interface GameVocabulary {
  paint: boolean;
  group: boolean;
  stream: boolean;
  lemma: "welford" | "online-softmax" | null;
}

export interface GameLevel {
  id: GameLevelId;
  exercise: 1 | 3 | 8 | 9;
  title: string;
  rung: string;
  framing: string;
  objective: string;
  vocabulary: GameVocabulary;
  hints: [string, string];
  baseline: NcdTerm;
  baselineCost: NapkinCost;
  target: { h: number; m?: number };
  solvedCost: NapkinCost;
  lemmaTitle?: string;
}

type PipelineConfig = {
  id: string;
  name: string;
  size: number;
  first: Pick<NcdBox, "id" | "label" | "kind" | "streamability">;
  middle: Pick<NcdBox, "id" | "label" | "kind" | "streamability">;
  last: Pick<NcdBox, "id" | "label" | "kind" | "streamability">;
  extraInputs?: Array<{
    id: string;
    label: string;
    consumer: "first" | "last";
  }>;
};

function copy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function decomposed(axisId: string, head: string, body: string) {
  return {
    kind: "decomposed" as const,
    axes: [{ axisId, head, body, accumulatorWireIds: [] }],
  };
}

function pipelineTerm(config: PipelineConfig): NcdTerm {
  const extras = config.extraInputs ?? [];
  const wires = [
    { id: "input", label: "X", axisIds: ["r"], elementBytes: 4 },
    ...extras.map((item) => ({
      id: item.id,
      label: item.label,
      axisIds: ["r"],
      elementBytes: 4,
    })),
    { id: "mid1", label: "A", axisIds: ["r"], elementBytes: 4 },
    { id: "mid2", label: "B", axisIds: ["r"], elementBytes: 4 },
    { id: "output", label: "Y", axisIds: ["r"], elementBytes: 4 },
  ];
  const firstExtras = extras
    .filter((item) => item.consumer === "first")
    .map((item) => item.id);
  const lastExtras = extras
    .filter((item) => item.consumer === "last")
    .map((item) => item.id);
  return {
    schemaVersion: "ncd-term-1-proposal",
    id: config.id,
    name: config.name,
    semantic: {
      axes: [{ id: "r", label: "r", size: config.size }],
      wires,
      boxes: [
        {
          ...config.first,
          column: 1,
          inputWireIds: ["input", ...firstExtras],
          outputWireIds: ["mid1"],
        },
        {
          ...config.middle,
          column: 3,
          inputWireIds: ["mid1"],
          outputWireIds: ["mid2"],
        },
        {
          ...config.last,
          column: 5,
          inputWireIds: ["mid2", ...lastExtras],
          outputWireIds: ["output"],
        },
      ],
      columns: [
        { id: "inputs", index: 0, label: "inputs at ℓ0" },
        { id: "pass-1", index: 1, label: config.first.label },
        { id: "boundary-1", index: 2, label: "materialize A" },
        { id: "pass-2", index: 3, label: config.middle.label },
        { id: "boundary-2", index: 4, label: "materialize B" },
        { id: "pass-3", index: 5, label: config.last.label },
        { id: "output", index: 6, label: "output at ℓ0" },
      ],
      tupleGroups: extras.length
        ? [
            {
              id: "inputs",
              label: "inputs",
              wireIds: ["input", ...extras.map((item) => item.id)],
            },
          ]
        : [],
    },
    decorations: {
      residency: [
        { wireId: "input", column: 0, level: "l0" },
        { wireId: "input", column: 1, level: "l1" },
        ...extras.flatMap((item) => [
          { wireId: item.id, column: 0, level: "l0" as const },
          {
            wireId: item.id,
            column: item.consumer === "first" ? 1 : 5,
            level: "l1" as const,
          },
        ]),
        { wireId: "mid1", column: 1, level: "l1" },
        { wireId: "mid1", column: 2, level: "l0" },
        { wireId: "mid1", column: 3, level: "l1" },
        { wireId: "mid2", column: 3, level: "l1" },
        { wireId: "mid2", column: 4, level: "l0" },
        { wireId: "mid2", column: 5, level: "l1" },
        { wireId: "output", column: 5, level: "l1" },
        { wireId: "output", column: 6, level: "l0" },
      ],
      partitions: [],
      divisibility: [{ axisId: "r", multiple: 32, reason: "row tile" }],
      admittedLemmas: [],
    },
  };
}

const chain = pipelineTerm({
  id: "game-bias-gelu-residual",
  name: "Unfused bias + GELU + residual",
  size: 1024,
  first: {
    id: "bias",
    label: "+ bias",
    kind: "elementwise",
    streamability: decomposed("r", "no state", "add bias element"),
  },
  middle: {
    id: "gelu",
    label: "GELU",
    kind: "elementwise",
    streamability: decomposed("r", "no state", "apply GELU element"),
  },
  last: {
    id: "residual",
    label: "+ residual",
    kind: "elementwise",
    streamability: decomposed("r", "no state", "add residual element"),
  },
  extraInputs: [
    { id: "bias-in", label: "bias", consumer: "first" },
    { id: "residual-in", label: "R", consumer: "last" },
  ],
});

const layernorm = pipelineTerm({
  id: "game-layernorm-three-pass",
  name: "LayerNorm — mean, variance, normalize",
  size: 1024,
  first: {
    id: "mean",
    label: "mean pass",
    kind: "reduce-mean",
    streamability: decomposed(
      "r",
      "initialize sum,count",
      "accumulate one row block",
    ),
  },
  middle: {
    id: "variance",
    label: "variance pass",
    kind: "reduce-variance",
    streamability: {
      kind: "none",
      reason:
        "variance depends on the completed mean, so the ordinary three-pass form needs the whole row",
    },
  },
  last: {
    id: "normalize",
    label: "normalize pass",
    kind: "normalize",
    streamability: decomposed(
      "r",
      "receive final moments",
      "normalize one row block",
    ),
  },
});

const softmax = pipelineTerm({
  id: "game-softmax-three-pass",
  name: "Softmax — max, exponential sum, normalize",
  size: 2048,
  first: {
    id: "maximum",
    label: "max pass",
    kind: "reduce-max",
    streamability: decomposed("r", "initialize m=-∞", "update block maximum"),
  },
  middle: {
    id: "softmax-sum",
    label: "exp-sum pass",
    kind: "exp-sum",
    streamability: {
      kind: "none",
      reason:
        "the exponential sum is expressed relative to a maximum unavailable until the whole row is seen",
    },
  },
  last: {
    id: "softmax-normalize",
    label: "normalize pass",
    kind: "softmax-normalize",
    streamability: decomposed(
      "r",
      "receive final m,ℓ",
      "normalize one streamed block",
    ),
  },
});

export function welfordLemma(term: NcdTerm): LemmaRelabeling {
  const box = term.semantic.boxes.find((item) => item.id === "variance");
  if (!box) throw new Error("LayerNorm term has no variance pass");
  return {
    op: "lemma",
    boxId: box.id,
    lemmaId: "welford-running-moments",
    before: {
      label: box.label,
      kind: box.kind,
      streamability: copy(box.streamability),
      inspection: box.inspection,
    },
    after: {
      label: "Welford moments",
      kind: "welford",
      streamability: decomposed(
        "r",
        "initialize count=0, μ=0, M2=0",
        "merge a block into running moments",
      ),
      inspection: {
        title: "Welford carried state",
        states: [
          {
            symbol: "μ",
            label: "running mean",
            explanation:
              "The mean of every value consumed so far; it moves as each block arrives.",
          },
          {
            symbol: "M2",
            label: "running squared deviation",
            explanation:
              "The accumulated squared distance from the moving mean; variance is M2/count at the end.",
          },
        ],
        correction: {
          expression: "δ²·nₐnᵦ/(nₐ+nᵦ)",
          explanation:
            "This merge correction accounts for the distance between two partial means.",
        },
      },
    },
    add: true,
  };
}

export function softmaxGameLemma(term: NcdTerm): LemmaRelabeling {
  const box = term.semantic.boxes.find((item) => item.id === "softmax-sum");
  if (!box) throw new Error("Softmax game term has no exponential-sum pass");
  return {
    op: "lemma",
    boxId: box.id,
    lemmaId: "online-softmax-rescaling",
    before: {
      label: box.label,
      kind: box.kind,
      streamability: copy(box.streamability),
      inspection: box.inspection,
    },
    after: {
      label: "online max + sum",
      kind: "online-softmax",
      streamability: decomposed(
        "r",
        "initialize m=-∞, ℓ=0",
        "update m, rescale ℓ, add block exponentials",
      ),
      inspection: {
        title: "Online-softmax carried state",
        states: [
          {
            symbol: "m",
            label: "running maximum",
            explanation: "Largest score seen in all streamed blocks so far.",
          },
          {
            symbol: "ℓ",
            label: "running normalizer",
            explanation:
              "Exponentials accumulated in the coordinate system of the current maximum.",
          },
        ],
        correction: {
          expression: "exp(m_old − m_new)",
          explanation:
            "Rescales every earlier contribution when a new block raises the running maximum.",
        },
      },
    },
    add: true,
  };
}

function recolorBoth(term: NcdTerm): NcdTerm {
  let next = term;
  for (const [wireId, column] of [
    ["mid1", 2],
    ["mid2", 4],
  ] as const) {
    next = applyMove(next, {
      op: "recolor",
      wireId,
      column,
      before: "l0",
      after: "l1",
    });
  }
  return next;
}

function streamRow(term: NcdTerm, size: number): NcdTerm {
  const after: PartitionDecoration = {
    axisId: "r",
    kind: "stream",
    size,
    label: "s_r",
  };
  return applyMove(term, { op: "partition", axisId: "r", after });
}

function slack(value: number): number {
  return Math.ceil(value * 1.1);
}

function makeLevel(
  definition: Omit<GameLevel, "baselineCost" | "solvedCost" | "target">,
  solved: NcdTerm,
  targetAxes: "h" | "hm",
): GameLevel {
  const baselineCost = napkinCost(definition.baseline);
  const solvedCost = napkinCost(solved);
  return {
    ...definition,
    baselineCost,
    solvedCost,
    target: {
      h: slack(solvedCost.transferByLevel.l1),
      m: targetAxes === "hm" ? slack(solvedCost.memoryByLevel.l1) : undefined,
    },
  };
}

const solvedChain = recolorBoth(cloneTerm(chain));
const solvedLayernorm = streamRow(
  recolorBoth(applyMove(cloneTerm(layernorm), welfordLemma(layernorm))),
  128,
);
const solvedSoftmax = streamRow(
  recolorBoth(applyMove(cloneTerm(softmax), softmaxGameLemma(softmax))),
  128,
);
const attention = attentionFixture as NcdTerm;
const solvedAttention = deriveFlashAttention(attention);

export const GAME_LEVELS: GameLevel[] = [
  makeLevel(
    {
      id: "fuse-chain",
      exercise: 1,
      title: "Fuse the chain",
      rung: "Rung 0 · traffic",
      framing:
        "Three cheap functions are paying for four unnecessary global-memory round trips.",
      objective:
        "Keep the intermediates below ℓ0 by painting away both materialization boundaries.",
      vocabulary: { paint: true, group: false, stream: false, lemma: null },
      hints: [
        "Look for magenta saves immediately followed by teal loads of the same array.",
        "Select the ℓ1 brush and paint the A and B materialization columns.",
      ],
      baseline: chain,
    },
    solvedChain,
    "h",
  ),
  makeLevel(
    {
      id: "layernorm",
      exercise: 3,
      title: "Carry the moments",
      rung: "Rung 5 preview · first lemma",
      framing:
        "LayerNorm crosses the row three times because variance waits for a completed mean.",
      objective:
        "Turn three passes into one streamed row program without exceeding the H/M budgets.",
      vocabulary: { paint: true, group: false, stream: true, lemma: "welford" },
      hints: [
        "The stream chip is the natural move; let the refusal name the dependency that blocks it.",
        "After earning Welford, inspect μ and M2, stream r=128, then remove both ℓ0 boundaries.",
      ],
      baseline: layernorm,
      lemmaTitle: "Welford running moments",
    },
    solvedLayernorm,
    "hm",
  ),
  makeLevel(
    {
      id: "softmax",
      exercise: 8,
      title: "Cross the lemma wall",
      rung: "Rung 5 · online streaming",
      framing:
        "Three-pass softmax stores a full row because max and sum appear to require the future.",
      objective:
        "Earn online softmax, inspect its carried state, and reach a one-pass streamed budget.",
      vocabulary: {
        paint: true,
        group: false,
        stream: true,
        lemma: "online-softmax",
      },
      hints: [
        "Try streaming r first; the jam identifies which quantity still depends on the whole row.",
        "Apply the earned lemma, inspect m and ℓ, stream r=128, and fuse both boundaries.",
      ],
      baseline: softmax,
      lemmaTitle: "Online-softmax rescaling",
    },
    solvedSoftmax,
    "hm",
  ),
  makeLevel(
    {
      id: "attention",
      exercise: 9,
      title: "Assemble FlashAttention",
      rung: "Capstone · composition",
      framing:
        "Eager attention materializes two N×N intermediates between three dispatches.",
      objective:
        "Compose the fusion, tiling, streaming, and lemma moves you have already learned.",
      vocabulary: {
        paint: true,
        group: true,
        stream: true,
        lemma: "online-softmax",
      },
      hints: [
        "Scores and probabilities are the expensive ℓ0 boundaries; streaming x exposes the familiar softmax wall.",
        "Earn the lemma, fuse S and P, group q=64, then stream x=32.",
      ],
      baseline: attention,
      lemmaTitle: "Online-softmax rescaling",
    },
    solvedAttention,
    "hm",
  ),
];

export function levelById(id: GameLevelId): GameLevel {
  const level = GAME_LEVELS.find((item) => item.id === id);
  if (!level) throw new Error(`Unknown NCD game level '${id}'`);
  return level;
}

export function lemmaMoveForLevel(
  level: GameLevel,
  term: NcdTerm,
): LemmaRelabeling {
  if (level.id === "layernorm") return welfordLemma(term);
  if (level.id === "softmax") return softmaxGameLemma(term);
  return onlineSoftmaxLemma(term);
}
