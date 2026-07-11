<script lang="ts">
  import { Gauge, LockKeyhole, Redo2, Undo2 } from "@lucide/svelte";
  import { scheduleStateHash } from "../../cost-model";
  import type {
    DeviceModel,
    ScheduleHistoryPoint,
    ScheduleState,
    StaticCost,
  } from "../../schedule-state";
  import LoopTree from "./LoopTree.svelte";

  type Props = {
    state: ScheduleState;
    cost: StaticCost;
    device: DeviceModel;
    history: ScheduleHistoryPoint[];
    historyCursor: number;
    structuralCosts: Array<{ label: string; cost: number }>;
    bindingNote: string;
    onEdit: (label: string, edit: (state: ScheduleState) => void) => void;
    onUndo: () => void;
    onRedo: () => void;
    onDeviceChange: (device: DeviceModel) => void;
  };

  let {
    state: schedule,
    cost,
    device,
    history,
    historyCursor,
    structuralCosts,
    bindingNote,
    onEdit,
    onUndo,
    onRedo,
    onDeviceChange,
  }: Props = $props();

  let benchNotice = $state("Awaiting engine benchmark channel.");

  const trace = $derived([
    ...history.map((point) => ({ label: point.label, value: point.cost.predictedMs })),
    ...structuralCosts.map((point) => ({ label: point.label, value: point.cost })),
  ]);
  const tracePoints = $derived.by(() => {
    if (!trace.length) return "";
    const values = trace.map((point) => point.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || Math.max(max, 1) * 0.05;
    return values
      .map((value, index) => {
        const x = values.length === 1 ? 50 : (index / (values.length - 1)) * 100;
        const y = 28 - ((value - min) / span) * 24;
        return `${x},${y}`;
      })
      .join(" ");
  });

  function fieldClass(): string {
    return "h-control w-20 border border-input bg-card px-1 type-value focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";
  }

  function selectClass(): string {
    return "h-control border border-input bg-card px-1 type-value focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";
  }

  function setTile(key: string, value: number): void {
    onEdit(`tile.${key} → ${value}`, (next) => {
      next.decorations.tileSizes[key] = Math.max(1, Math.floor(value));
      if (next.algorithm.workload.kind === "matmul") {
        const threadM = next.decorations.tileSizes.threadM;
        const threadN = next.decorations.tileSizes.threadN;
        next.decorations.workgroup = {
          x: Math.max(1, Math.floor(next.decorations.tileSizes.n / threadN)),
          y: Math.max(1, Math.floor(next.decorations.tileSizes.m / threadM)),
          z: 1,
        };
      }
    });
  }

  function setWorkgroup(axis: "x" | "y" | "z", value: number): void {
    onEdit(`workgroup.${axis} → ${value}`, (next) => {
      next.decorations.workgroup[axis] = Math.max(1, Math.floor(value));
    });
  }

  function setDevice(
    key: "peakBandwidthGBs" | "peakFlopsTFLOPs",
    value: number,
  ): void {
    onDeviceChange({ ...device, [key]: Math.max(0.001, value) });
  }

  function formatBytes(bytes: number): string {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
    return `${bytes} B`;
  }

  function formatFlops(flops: number): string {
    if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOP`;
    if (flops >= 1e9) return `${(flops / 1e9).toFixed(2)} GFLOP`;
    return `${flops.toFixed(0)} FLOP`;
  }
</script>

<section class="stack-field border-t border-border bg-background pad-box" aria-label="Intra-island workbench">
  <div class="flex flex-wrap items-center justify-between gap-2">
    <div class="stack-tight">
      <div class="flex items-center gap-1.5">
        <h2 class="type-heading">{schedule.name}</h2>
        <span class={schedule.authored ? "type-tag text-warning-strong" : "type-tag text-success"}>
          {schedule.authored ? "AUTHORED" : "DERIVED"}
        </span>
      </div>
      <p class="type-body text-muted-foreground">{bindingNote}</p>
    </div>
    <div class="flex items-center gap-1">
      <span class="type-value">{scheduleStateHash(schedule)}</span>
      <button class="inline-flex h-control items-center gap-1 border border-border bg-card px-1.5 type-button hover:bg-muted active:bg-border/50 disabled:opacity-50" disabled={historyCursor <= 0} onclick={onUndo}>
        <Undo2 size={11} /> State
      </button>
      <button class="inline-flex h-control items-center gap-1 border border-border bg-card px-1.5 type-button hover:bg-muted active:bg-border/50 disabled:opacity-50" disabled={historyCursor >= history.length - 1} onclick={onRedo}>
        <Redo2 size={11} /> State
      </button>
    </div>
  </div>

  <div class="grid grid-cols-[minmax(18rem,1.15fr)_minmax(17rem,0.85fr)] items-start gap-1 max-[1050px]:grid-cols-1">
    <div class="stack-group">
      <section class="stack-field border border-border">
        <header class="flex items-center justify-between gap-2 border-b border-border bg-panel px-1 py-0.5">
          <h3 class="type-title">Macro skeleton</h3>
          <span class="type-tag">{schedule.skeleton.visibility}</span>
        </header>
        {#if schedule.skeleton.visibility === "opaque"}
          <div class="stack-field pad-box">
            <div class="flex items-center gap-1 border border-warning bg-warning/10 pad-box text-warning-strong">
              <LockKeyhole size={13} />
              <span class="type-label">Authored — not yet re-derived</span>
            </div>
            <p class="prose">Macro moves are locked because this kernel has no re-derived skeleton. Decorations remain editable.</p>
            <dl class="grid grid-cols-[auto_1fr] gap-1">
              <dt class="type-label">Kernel</dt><dd class="type-code">{schedule.skeleton.kernelRef}</dd>
              <dt class="type-label">Refusal</dt><dd class="type-body text-muted-foreground">{schedule.skeleton.reason}</dd>
            </dl>
          </div>
        {:else}
          <div class="stack-field pad-box">
            <div class="stack-tight">
              <span class="type-label">Loop nest</span>
              <LoopTree loops={schedule.skeleton.loopNest} />
            </div>
            <div class="stack-tight">
              <span class="type-label">Staging edges</span>
              <div class="stack-tight">
                {#each schedule.skeleton.stagingEdges as edge}
                  <div class="grid grid-cols-[auto_1fr_auto] items-center gap-1 border border-border bg-card pad-box">
                    <span class="type-tag">{edge.operand}</span>
                    <span class="type-code">{edge.from} → {edge.to}</span>
                    <span class="type-tag">{edge.synchronization}</span>
                    <span class="type-body text-muted-foreground col-span-3">{edge.scope}</span>
                  </div>
                {/each}
              </div>
            </div>
            <div class="stack-tight">
              <span class="type-label">Role partition</span>
              <div class="stack-tight">
                {#each schedule.skeleton.rolePartition as role}
                  <div class="grid grid-cols-[auto_1fr] gap-1 border border-border bg-card pad-box">
                    <span class="type-tag">{role.role}</span>
                    <span class="type-body">{role.participants}</span>
                    <span class="type-body text-muted-foreground col-span-2">{role.responsibility}</span>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        {/if}
      </section>

      <section class="stack-field border border-border">
        <header class="border-b border-border bg-panel px-1 py-0.5"><h3 class="type-title">Decorations</h3></header>
        <div class="stack-field pad-box">
          <div class="grid grid-cols-[repeat(auto-fit,minmax(9rem,1fr))] gap-1">
            {#each Object.entries(schedule.decorations.tileSizes) as [key, value]}
              <label class="stack-tight">
                <span class="type-label">Tile {key}</span>
                <input class={fieldClass()} type="number" min="1" value={value} onchange={(event) => setTile(key, event.currentTarget.valueAsNumber)} />
              </label>
            {/each}
          </div>
          <div class="grid grid-cols-[repeat(auto-fit,minmax(8rem,1fr))] gap-1">
            {#each ["x", "y", "z"] as axis}
              <label class="stack-tight">
                <span class="type-label">Workgroup {axis}</span>
                <input class={fieldClass()} type="number" min="1" value={schedule.decorations.workgroup[axis as "x" | "y" | "z"]} onchange={(event) => setWorkgroup(axis as "x" | "y" | "z", event.currentTarget.valueAsNumber)} />
              </label>
            {/each}
            <label class="stack-tight">
              <span class="type-label">Vector width</span>
              <select class={selectClass()} value={String(schedule.decorations.vectorWidth)} onchange={(event) => onEdit(`vector width → ${event.currentTarget.value}`, (next) => { next.decorations.vectorWidth = Number(event.currentTarget.value) as 1 | 2 | 4; })}>
                <option value="1">1</option><option value="2">2</option><option value="4">4</option>
              </select>
            </label>
            <label class="stack-tight">
              <span class="type-label">Unroll</span>
              <input class={fieldClass()} type="number" min="1" value={schedule.decorations.unrollFactor} onchange={(event) => onEdit(`unroll → ${event.currentTarget.value}`, (next) => { next.decorations.unrollFactor = Math.max(1, event.currentTarget.valueAsNumber); })} />
            </label>
            <label class="stack-tight opacity-50" title="WGSL realizer pins pipeline depth to 1">
              <span class="type-label">Pipeline depth</span>
              <select class={selectClass()} disabled value={String(schedule.decorations.pipelineDepth)}><option value="1">1 · WGSL pinned</option></select>
            </label>
          </div>
          <div class="grid grid-cols-[repeat(auto-fit,minmax(10rem,1fr))] gap-1">
            <label class="stack-tight opacity-50" title="WGSL v1 exposes the global + workgroup-shared set">
              <span class="type-label">Memory spaces</span>
              <select class={selectClass()} disabled><option>global + shared</option></select>
            </label>
            <label class="stack-tight opacity-50" title="WGSL v1 pins synchronization to workgroup scope">
              <span class="type-label">Sync scope</span>
              <select class={selectClass()} disabled><option>workgroup</option></select>
            </label>
            <label class="stack-tight">
              <span class="type-label">Thread hierarchy</span>
              <select
                class={selectClass()}
                value={schedule.decorations.threadHierarchy.includes("subgroup") ? "subgroup" : "workgroup"}
                onchange={(event) => onEdit(`thread hierarchy → ${event.currentTarget.value}`, (next) => {
                  next.decorations.threadHierarchy = event.currentTarget.value === "subgroup"
                    ? ["invocation", "subgroup", "workgroup"]
                    : ["invocation", "workgroup"];
                })}
              >
                <option value="workgroup">invocation + workgroup</option>
                <option value="subgroup" disabled={!device.subgroupsSupported}>+ subgroup {device.subgroupsSupported ? "" : "(unsupported)"}</option>
              </select>
            </label>
          </div>
          <div class="stack-tight">
            <span class="type-label">Operand residency</span>
            <div class="grid grid-cols-[repeat(auto-fit,minmax(10rem,1fr))] gap-1">
              {#each Object.entries(schedule.decorations.operandResidency) as [operand, residency]}
                <label class="stack-tight">
                  <span class="type-tag">{operand}</span>
                  <select class={selectClass()} value={residency} onchange={(event) => onEdit(`${operand} residency → ${event.currentTarget.value}`, (next) => { next.decorations.operandResidency[operand] = event.currentTarget.value as "global" | "workgroup-shared" | "register"; })}>
                    <option value="global">global</option>
                    <option value="workgroup-shared">workgroup-shared</option>
                    <option value="register">register</option>
                  </select>
                </label>
              {/each}
            </div>
          </div>
          <dl class="grid grid-cols-[auto_1fr] gap-1 border border-border bg-card pad-box">
            <dt class="type-label">Realizer</dt><dd class="type-code">{schedule.realizerId}</dd>
            <dt class="type-label">Atoms</dt><dd class="type-body text-muted-foreground">{schedule.atoms.length ? schedule.atoms.map((atom) => atom.kind).join(", ") : "none"}</dd>
            <dt class="type-label">Lemmas</dt><dd class="type-body text-muted-foreground">{schedule.admittedLemmas.length ? schedule.admittedLemmas.map((lemma) => lemma.id).join(", ") : "none"}</dd>
          </dl>
        </div>
      </section>
    </div>

    <div class="stack-group">
      <section class="stack-field border border-border">
        <header class="flex items-center justify-between gap-2 border-b border-border bg-panel px-1 py-0.5">
          <h3 class="type-title">Static performance</h3>
          <span class="type-tag">LIVE · {device.source}</span>
        </header>
        <div class="stack-field pad-box">
          <div class="grid grid-cols-2 gap-1">
            <label class="stack-tight"><span class="type-label">Peak bandwidth</span><span class="flex items-center gap-1"><input class={fieldClass()} type="number" min="0.001" step="10" value={device.peakBandwidthGBs} onchange={(event) => setDevice("peakBandwidthGBs", event.currentTarget.valueAsNumber)} /><span class="type-tag">GB/S</span></span></label>
            <label class="stack-tight"><span class="type-label">Peak compute</span><span class="flex items-center gap-1"><input class={fieldClass()} type="number" min="0.001" step="0.5" value={device.peakFlopsTFLOPs} onchange={(event) => setDevice("peakFlopsTFLOPs", event.currentTarget.valueAsNumber)} /><span class="type-tag">TFLOP/S</span></span></label>
          </div>
          <div class="grid grid-cols-[1fr_auto] gap-1 border border-border bg-card pad-box">
            <span class="type-label">Shared memory</span><span class={cost.sharedMemoryUtilization > 1 ? "type-value text-destructive-strong" : "type-value"}>{formatBytes(cost.sharedMemoryBytes)} / {formatBytes(device.maxComputeWorkgroupStorageSize)}</span>
            <span class="type-label">Storage pressure</span><span class="type-value">{(cost.sharedMemoryUtilization * 100).toFixed(1)}%</span>
            <span class="type-label">Workgroup</span><span class={cost.workgroupThreads > device.maxComputeInvocationsPerWorkgroup ? "type-value text-destructive-strong" : "type-value"}>{cost.workgroupThreads} / {device.maxComputeInvocationsPerWorkgroup}</span>
            <span class="type-label">Resident WG proxy</span><span class="type-value">{cost.residentWorkgroupsProxy} · {cost.occupancyLimiter}</span>
            <span class="type-label">Occupancy proxy</span><span class="type-value">{(cost.occupancyProxy * 100).toFixed(1)}%</span>
            <span class="type-label">Bytes moved</span><span class="type-value">{formatBytes(cost.bytesMoved)}</span>
            <span class="type-label">Arithmetic</span><span class="type-value">{formatFlops(cost.flops)}</span>
            <span class="type-label">Intensity</span><span class="type-value">{cost.arithmeticIntensity.toFixed(2)} FLOP/B</span>
            <span class="type-label">Ridge point</span><span class="type-value">{cost.ridgePoint.toFixed(2)} FLOP/B</span>
            <span class="type-label">Roofline</span><span class={cost.rooflineBound === "bandwidth" ? "type-tag text-warning-strong" : "type-tag text-success"}>{cost.rooflineBound}</span>
            <span class="type-label">Attainable</span><span class="type-value">{cost.attainableTFLOPs.toFixed(2)} TFLOP/S</span>
            <span class="type-label">Static floor</span><span class="type-value">{cost.predictedMs.toFixed(4)} ms</span>
          </div>
          {#if cost.sharedMemoryUtilization > 1 || cost.workgroupThreads > device.maxComputeInvocationsPerWorkgroup}
            <p class="border border-destructive bg-destructive/10 pad-box type-body text-destructive-strong">Realizer refusal expected: this state exceeds an adapter limit.</p>
          {/if}
          <p class="type-fine text-muted-foreground">Adapter supplies per-workgroup limits. Resident slots, per-CU shared memory, bandwidth, and FLOPS are documented fallback/device-model constants.</p>
        </div>
      </section>

      <section class="stack-field border border-border">
        <header class="flex items-center justify-between gap-2 border-b border-border bg-panel px-1 py-0.5">
          <h3 class="type-title">Measured performance</h3>
          <span class="type-tag text-muted-foreground">AWAITING ENGINE</span>
        </header>
        <div class="stack-field pad-box">
          <div class="flex items-center gap-1 border border-border bg-card pad-box">
            <Gauge size={13} />
            <span class="type-body">No timestamp-query benchmark response.</span>
          </div>
          <dl class="grid grid-cols-[auto_1fr] gap-1">
            <dt class="type-label">State hash</dt><dd class="type-value">{scheduleStateHash(schedule)}</dd>
            <dt class="type-label">Protocol</dt><dd class="type-code">schedule.bench.request.v1</dd>
            <dt class="type-label">Result</dt><dd class="type-body text-muted-foreground">median ms + raw samples</dd>
          </dl>
          <button class="h-control border border-border bg-card px-1.5 type-button hover:bg-muted active:bg-border/50" onclick={() => (benchNotice = "Awaiting engine: bench request contract is declared but no RPC is connected.")}>Request measurement</button>
          <p class="type-body text-muted-foreground">{benchNotice}</p>
        </div>
      </section>

      <section class="stack-field border border-border">
        <header class="flex items-center justify-between gap-2 border-b border-border bg-panel px-1 py-0.5"><h3 class="type-title">Cost-annotated history</h3><span class="type-value">{trace.length} states</span></header>
        <div class="stack-field pad-box">
          <svg class="h-8 w-full border border-border bg-card text-primary" viewBox="0 0 100 32" preserveAspectRatio="none" role="img" aria-label="Predicted duration history">
            <polyline points={tracePoints} fill="none" stroke="currentColor" stroke-width="1" vector-effect="non-scaling-stroke" />
            {#each trace as point, index}
              {@const x = trace.length === 1 ? 50 : (index / (trace.length - 1)) * 100}
              <line x1={x} x2={x} y1="28" y2="31" stroke="currentColor" stroke-width="1" vector-effect="non-scaling-stroke" />
            {/each}
          </svg>
          <div class="max-h-28 overflow-y-auto border border-border">
            {#each trace as point, index}
              <div class={`grid grid-cols-[auto_1fr_auto] gap-1 border-b border-border pad-box last:border-b-0 ${index === historyCursor ? "bg-primary/12" : "bg-background"}`}>
                <span class="type-value">{index}</span><span class="truncate type-body">{point.label}</span><span class="type-value">{point.value.toFixed(4)} ms</span>
              </div>
            {/each}
          </div>
          <p class="type-fine text-muted-foreground">Lower is warmer. Decoration states and structural partition entries share the same static-cost trace.</p>
        </div>
      </section>
    </div>
  </div>
</section>
