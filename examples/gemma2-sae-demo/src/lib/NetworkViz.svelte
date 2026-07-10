<script lang="ts">
  import { twMerge } from "tailwind-merge";
  import Panel from "./components/primitives/Panel.svelte";

  /**
   * Generic (architecture-agnostic) network view, left-to-right by depth.
   * Columns come from the numeric segment in dotted tensor names
   * (`model.layers.12.mlp.gate_proj.weight` → column 12); un-numbered
   * tensors (embeddings, final norm, lm_head) get their own columns ordered
   * by file position. Each block is one parameter tensor, height ∝ √params,
   * colored by leaf kind, filling bottom-up as its bytes stream to the GPU.
   */

  type ManifestTensor = {
    name: string;
    shape: number[];
    elems: number;
    dtype: string;
    skipped: boolean;
  };

  let {
    tensors,
    fill,
  }: {
    tensors: ManifestTensor[];
    /** name → 0..1 loaded fraction (1 = resident on GPU). */
    fill: Record<string, number>;
  } = $props();

  const CHART_CLASSES = ["bg-chart-1", "bg-chart-2", "bg-chart-3", "bg-chart-4", "bg-chart-5"];
  const MAX_BLOCK_PX = 64;
  const MIN_BLOCK_PX = 3;

  /** Leaf kind: trailing name segments minus indices/"weight" — e.g. "mlp.gate_proj". */
  function leafKind(name: string): string {
    const parts = name.split(".").filter((p) => !/^\d+$/.test(p) && p !== "weight" && p !== "model");
    return parts.slice(-2).join(".");
  }

  const layout = $derived.by(() => {
    type Col = { key: string; label: string; num: number | null; minIdx: number; items: ManifestTensor[] };
    const cols = new Map<string, Col>();
    tensors.forEach((t, idx) => {
      const m = t.name.match(/\.(\d+)\./);
      const num = m ? Number(m[1]) : null;
      const key = num !== null ? `#${num}` : leafKind(t.name);
      let col = cols.get(key);
      if (!col) {
        col = { key, label: num !== null ? String(num) : leafKind(t.name), num, minIdx: idx, items: [] };
        cols.set(key, col);
      }
      col.items.push(t);
    });
    const all = [...cols.values()];
    // Numbered columns sort by their number, anchored where the first
    // numbered column appears in file order; standalone columns keep file order.
    const numberedBase = Math.min(...all.filter((c) => c.num !== null).map((c) => c.minIdx), Infinity);
    const orderKey = (c: Col) => (c.num !== null ? numberedBase + c.num * 1e-3 : c.minIdx);
    all.sort((a, b) => orderKey(a) - orderKey(b));
    for (const c of all) c.items.sort((x, y) => leafKind(x.name).localeCompare(leafKind(y.name)));

    // Stable leaf → color assignment (first-seen order).
    const kinds: string[] = [];
    for (const t of tensors) {
      const k = leafKind(t.name);
      if (!kinds.includes(k)) kinds.push(k);
    }
    const colorOf = (name: string) => CHART_CLASSES[kinds.indexOf(leafKind(name)) % CHART_CLASSES.length];

    const maxElems = Math.max(...tensors.map((t) => t.elems), 1);
    const blockPx = (elems: number) =>
      Math.max(MIN_BLOCK_PX, Math.round(Math.sqrt(elems / maxElems) * MAX_BLOCK_PX));

    return { cols: all, kinds, colorOf, blockPx };
  });

  const loadedCount = $derived(tensors.filter((t) => (fill[t.name] ?? 0) >= 1).length);
</script>

<Panel title="Network" contentClass="p-1.5 stack-tight">
  <div class="flex items-end gap-px overflow-x-auto pb-0.5">
    {#each layout.cols as col (col.key)}
      <div class="flex shrink-0 flex-col justify-end gap-px" title={col.num !== null ? `layer ${col.num}` : col.label}>
        {#each col.items as t (t.name)}
          {@const f = fill[t.name] ?? 0}
          <div
            class={twMerge(
              "relative w-3.5 border border-border overflow-hidden",
              t.skipped && "opacity-30",
            )}
            style="height: {layout.blockPx(t.elems)}px"
            title={`${t.name}  [${t.shape.join("×")}]  ${t.dtype}${t.skipped ? "  (skipped: tied)" : ""}`}
          >
            <div
              class={twMerge("absolute inset-x-0 bottom-0", layout.colorOf(t.name), f < 1 && "opacity-60")}
              style="height: {Math.round(f * 100)}%"
            ></div>
          </div>
        {/each}
        <span
          class="h-3 overflow-hidden text-center type-label text-subtle-foreground"
          style="font-size: 8px"
          title={col.label}
        >
          {col.num !== null ? (col.num % 4 === 0 ? col.num : "") : col.label.replace(/[^a-z0-9]/gi, "").slice(0, 2)}
        </span>
      </div>
    {/each}
  </div>
  <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5">
    {#each layout.kinds as kind, i (kind)}
      <span class="flex items-center gap-1 type-label text-subtle-foreground">
        <span class={twMerge("inline-block h-2 w-2 border border-border", CHART_CLASSES[i % CHART_CLASSES.length])}></span>
        {kind}
      </span>
    {/each}
    <span class="ml-auto type-value text-muted-foreground">{loadedCount}/{tensors.length} tensors</span>
  </div>
</Panel>
