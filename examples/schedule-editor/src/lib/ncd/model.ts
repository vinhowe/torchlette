import type {
  LemmaRelabeling,
  NapkinCost,
  NcdAxis,
  NcdBox,
  NcdDiagramModel,
  NcdLevel,
  NcdMove,
  NcdTerm,
  NcdWire,
  PartitionDecoration,
  ProjectionResult,
} from "./types";

export function cloneTerm(term: NcdTerm): NcdTerm {
  return JSON.parse(JSON.stringify(term)) as NcdTerm;
}

function copy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export function toDiagram(term: NcdTerm): NcdDiagramModel {
  return {
    schemaVersion: "ncd-diagram-1",
    meta: {
      id: term.id,
      name: term.name,
      termSchemaVersion: term.schemaVersion,
    },
    axes: copy(term.semantic.axes),
    bundles: copy(term.semantic.wires),
    functions: copy(term.semantic.boxes),
    columns: copy(term.semantic.columns),
    tupleGroups: copy(term.semantic.tupleGroups),
    labels: copy(term.decorations),
  };
}

export function fromDiagram(diagram: NcdDiagramModel): NcdTerm {
  return {
    schemaVersion: diagram.meta.termSchemaVersion,
    id: diagram.meta.id,
    name: diagram.meta.name,
    semantic: {
      axes: copy(diagram.axes),
      wires: copy(diagram.bundles),
      boxes: copy(diagram.functions),
      columns: copy(diagram.columns),
      tupleGroups: copy(diagram.tupleGroups),
    },
    decorations: copy(diagram.labels),
  };
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export function termHash(term: NcdTerm): string {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(canonical(term))) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

export function axisById(term: NcdTerm, axisId: string): NcdAxis {
  const axis = term.semantic.axes.find((item) => item.id === axisId);
  if (!axis) throw new Error(`Unknown NCD axis '${axisId}'`);
  return axis;
}

export function wireById(term: NcdTerm, wireId: string): NcdWire {
  const wire = term.semantic.wires.find((item) => item.id === wireId);
  if (!wire) throw new Error(`Unknown NCD wire '${wireId}'`);
  return wire;
}

function boxUsesAxis(term: NcdTerm, box: NcdBox, axisId: string): boolean {
  return [...box.inputWireIds, ...box.outputWireIds].some((wireId) =>
    wireById(term, wireId).axisIds.includes(axisId),
  );
}

export function streamability(
  term: NcdTerm,
  axisId: string,
): { legal: boolean; reason: string } {
  const axis = axisById(term, axisId);
  const relevant = term.semantic.boxes.filter((box) =>
    boxUsesAxis(term, box, axisId),
  );
  if (!relevant.length) {
    return {
      legal: false,
      reason: `No function is mapped over axis ${axis.label}.`,
    };
  }
  for (const box of relevant) {
    if (box.streamability.kind === "none") {
      return {
        legal: false,
        reason: `${box.label} has no head/body decomposition: ${box.streamability.reason}`,
      };
    }
    if (!box.streamability.axes.some((item) => item.axisId === axisId)) {
      return {
        legal: false,
        reason: `${box.label} does not expose a head/body decomposition along ${axis.label}.`,
      };
    }
  }
  return {
    legal: true,
    reason: `${relevant.map((box) => box.label).join(" ∘ ")} is streamable along ${axis.label}.`,
  };
}

export function partitionLegality(
  term: NcdTerm,
  partition: PartitionDecoration,
): { legal: boolean; reason: string } {
  const axis = axisById(term, partition.axisId);
  if (!Number.isInteger(partition.size) || partition.size <= 0) {
    return {
      legal: false,
      reason: "Partition size must be a positive integer.",
    };
  }
  if (axis.size % partition.size !== 0) {
    return {
      legal: false,
      reason: `${axis.label}=${axis.size} is not divisible by ${partition.size}.`,
    };
  }
  const constraint = term.decorations.divisibility.find(
    (item) => item.axisId === partition.axisId,
  );
  if (constraint && partition.size % constraint.multiple !== 0) {
    return {
      legal: false,
      reason: `${partition.label} must be divisible by ${constraint.multiple} (${constraint.reason}).`,
    };
  }
  if (partition.kind === "stream") return streamability(term, partition.axisId);
  return {
    legal: true,
    reason: `${partition.label} is a legal group partition of ${axis.label}.`,
  };
}

function producer(term: NcdTerm, wireId: string): NcdBox | undefined {
  return term.semantic.boxes.find((box) => box.outputWireIds.includes(wireId));
}

function consumer(term: NcdTerm, wireId: string): NcdBox | undefined {
  return term.semantic.boxes.find((box) => box.inputWireIds.includes(wireId));
}

export function recolorLegality(
  term: NcdTerm,
  wireId: string,
  column: number,
  target: NcdLevel,
): { legal: boolean; reason: string } {
  const stateIndex = term.decorations.residency.findIndex(
    (item) => item.wireId === wireId && item.column === column,
  );
  if (stateIndex < 0)
    return { legal: false, reason: "Wire segment does not exist." };
  const current = term.decorations.residency[stateIndex];
  if (current.level === target) {
    return { legal: false, reason: `Segment is already at ${target}.` };
  }
  if (target === "l0") {
    return {
      legal: true,
      reason: "Unfuse restores an ℓ0 materialization boundary.",
    };
  }
  const states = term.decorations.residency
    .filter((item) => item.wireId === wireId)
    .sort((a, b) => a.column - b.column);
  const index = states.findIndex((item) => item.column === column);
  if (
    index <= 0 ||
    index >= states.length - 1 ||
    states[index - 1].level !== "l1" ||
    states[index + 1].level !== "l1"
  ) {
    return {
      legal: false,
      reason:
        "Only an interior ℓ0 materialization between two lower-level regions can fuse.",
    };
  }
  const left = producer(term, wireId);
  const right = consumer(term, wireId);
  if (!left || !right) {
    return {
      legal: false,
      reason: "Fusion boundary lacks a producer/consumer pair.",
    };
  }
  const wire = wireById(term, wireId);
  const common = wire.axisIds.find((axisId) => {
    if (left.streamability.kind !== "decomposed") return false;
    if (right.streamability.kind !== "decomposed") return false;
    return (
      left.streamability.axes.some((item) => item.axisId === axisId) &&
      right.streamability.axes.some((item) => item.axisId === axisId)
    );
  });
  if (!common) {
    const blocked = [left, right]
      .filter((box) => box.streamability.kind === "none")
      .map((box) =>
        box.streamability.kind === "none"
          ? `${box.label}: ${box.streamability.reason}`
          : box.label,
      )
      .join("; ");
    return {
      legal: false,
      reason:
        blocked ||
        `${left.label} and ${right.label} have no common stream decomposition.`,
    };
  }
  return {
    legal: true,
    reason: `Fusion theorem applies: ${left.label} and ${right.label} compose along ${axisById(term, common).label}.`,
  };
}

function partitionForAxis(
  term: NcdTerm,
  axisId: string,
): PartitionDecoration | undefined {
  return term.decorations.partitions.find((item) => item.axisId === axisId);
}

export function wireElementsAtLevel(
  term: NcdTerm,
  wire: NcdWire,
  level: NcdLevel,
): number {
  return wire.axisIds.reduce((product, axisId) => {
    const axis = axisById(term, axisId);
    const partition =
      level === "l0" ? undefined : partitionForAxis(term, axisId);
    return product * (partition?.size ?? axis.size);
  }, 1);
}

export function napkinCost(term: NcdTerm): NapkinCost {
  const transferByLevel: Record<NcdLevel, number> = { l0: 0, l1: 0 };
  const transferBytesByLevel: Record<NcdLevel, number> = { l0: 0, l1: 0 };
  const memoryByLevel: Record<NcdLevel, number> = { l0: 0, l1: 0 };
  const memoryBytesByLevel: Record<NcdLevel, number> = { l0: 0, l1: 0 };

  for (const wire of term.semantic.wires) {
    const states = term.decorations.residency
      .filter((item) => item.wireId === wire.id)
      .sort((a, b) => a.column - b.column);
    for (let index = 1; index < states.length; index += 1) {
      if (states[index - 1].level === states[index].level) continue;
      const lower: NcdLevel =
        states[index - 1].level === "l1" || states[index].level === "l1"
          ? "l1"
          : "l0";
      const elements = wireElementsAtLevel(term, wire, lower);
      transferByLevel[lower] += elements;
      transferBytesByLevel[lower] += elements * wire.elementBytes;
    }
  }

  for (const column of term.semantic.columns) {
    for (const level of ["l0", "l1"] as const) {
      let elements = 0;
      let bytes = 0;
      for (const state of term.decorations.residency.filter(
        (item) => item.column === column.index && item.level === level,
      )) {
        const wire = wireById(term, state.wireId);
        const count = wireElementsAtLevel(term, wire, level);
        elements += count;
        bytes += count * wire.elementBytes;
      }
      memoryByLevel[level] = Math.max(memoryByLevel[level], elements);
      memoryBytesByLevel[level] = Math.max(memoryBytesByLevel[level], bytes);
    }
  }

  return {
    transferByLevel,
    memoryByLevel,
    transferBytesByLevel,
    memoryBytesByLevel,
  };
}

export function applyMove(term: NcdTerm, move: NcdMove): NcdTerm {
  const next = cloneTerm(term);
  if (move.op === "partition") {
    next.decorations.partitions = next.decorations.partitions.filter(
      (item) => item.axisId !== move.axisId,
    );
    if (move.after) next.decorations.partitions.push(copy(move.after));
  } else if (move.op === "recolor") {
    const state = next.decorations.residency.find(
      (item) => item.wireId === move.wireId && item.column === move.column,
    );
    if (!state || state.level !== move.before) {
      throw new Error("Recolor precondition no longer matches the term");
    }
    state.level = move.after;
  } else {
    const box = next.semantic.boxes.find((item) => item.id === move.boxId);
    if (!box) throw new Error(`Unknown lemma box '${move.boxId}'`);
    box.label = move.after.label;
    box.kind = move.after.kind;
    box.streamability = copy(move.after.streamability);
    box.inspection = move.after.inspection
      ? copy(move.after.inspection)
      : undefined;
    next.decorations.admittedLemmas = next.decorations.admittedLemmas.filter(
      (item) => item !== move.lemmaId,
    );
    if (move.add) next.decorations.admittedLemmas.push(move.lemmaId);
  }
  return next;
}

export function inverseMove(move: NcdMove): NcdMove {
  if (move.op === "partition") {
    return { ...move, before: move.after, after: move.before };
  }
  if (move.op === "recolor") {
    return { ...move, before: move.after, after: move.before };
  }
  return {
    ...move,
    before: move.after,
    after: move.before,
    add: !move.add,
  };
}

export function onlineSoftmaxLemma(term: NcdTerm): LemmaRelabeling {
  const box = term.semantic.boxes.find((item) => item.id === "softmax");
  if (!box) throw new Error("Term has no softmax box");
  return {
    op: "lemma",
    boxId: box.id,
    lemmaId: "online-softmax-rescaling",
    before: {
      label: box.label,
      kind: box.kind,
      streamability: copy(box.streamability),
      inspection: box.inspection ? copy(box.inspection) : undefined,
    },
    after: {
      label: "online softmax",
      kind: "online-softmax",
      streamability: {
        kind: "decomposed",
        axes: [
          {
            axisId: "x",
            head: "initialize m=-∞, l=0, o=0",
            body: "update m,l and rescale o by exp(m_old-m_new)",
            accumulatorWireIds: ["scores", "probabilities"],
          },
        ],
      },
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
              "Sum of exponentials expressed relative to the current running maximum.",
          },
        ],
        correction: {
          expression: "exp(m_old − m_new)",
          explanation:
            "When a later block raises the maximum, this rescales every earlier contribution into the new reference frame.",
        },
      },
    },
    add: true,
  };
}

export function deriveProjection(term: NcdTerm): ProjectionResult {
  for (const partition of term.decorations.partitions) {
    const legal = partitionLegality(term, partition);
    if (!legal.legal) {
      return { ok: false, lines: [], reason: legal.reason };
    }
  }
  const groups = term.decorations.partitions.filter(
    (item) => item.kind === "group",
  );
  const streams = term.decorations.partitions.filter(
    (item) => item.kind === "stream",
  );
  const lines: string[] = ["dispatch schedule:"];
  let indent = "  ";
  for (const group of groups) {
    const axis = axisById(term, group.axisId);
    lines.push(
      `${indent}for ${axis.label}₀ in 0..${axis.size} step ${group.size}:`,
    );
    indent += "  ";
  }
  for (const stream of streams) {
    const axis = axisById(term, stream.axisId);
    const decompositions = term.semantic.boxes.flatMap((box) =>
      box.streamability.kind === "decomposed"
        ? box.streamability.axes
            .filter((item) => item.axisId === stream.axisId)
            .map((item) => `${box.label}: ${item.head}`)
        : [],
    );
    for (const head of decompositions) lines.push(`${indent}${head}`);
    lines.push(
      `${indent}for ${axis.label}₀ in 0..${axis.size} step ${stream.size}:`,
    );
    indent += "  ";
  }

  const ordered = [...term.semantic.boxes].sort((a, b) => a.column - b.column);
  for (const box of ordered) {
    if (streams.length && box.streamability.kind === "decomposed") {
      const bodies = box.streamability.axes
        .filter((item) =>
          streams.some((stream) => stream.axisId === item.axisId),
        )
        .map((item) => item.body);
      lines.push(
        `${indent}${box.label}  // ${bodies.join("; ") || "mapped body"}`,
      );
    } else {
      lines.push(`${indent}${box.label}`);
    }
  }
  if (!groups.length && !streams.length) {
    lines.push("  materialize every ℓ0 intermediate between boxes");
  }
  return { ok: true, lines };
}
