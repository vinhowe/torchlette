/**
 * The 4 per-token upload payloads for a seqLen=1 static-KV Qwen3 decode step,
 * byte-identical to what model.forward constructs (model.ts §forward). Keyed by
 * shape (all four shapes are distinct within a bucket) so the step-tape replay
 * can dress the skeleton's tensorFromArray nodes unambiguously. No side effects
 * — safe to import from any driver (docs/staged-execution-phase1.md §1c).
 */

type Cfg = { headDim: number; numKVHeads: number };

/**
 * Stable per-instance id for a StaticKV. The tape skeleton binds SPECIFIC KV
 * cache buffers (external inputs resolved from the recording step's planNodes),
 * so a skeleton is only valid for the KV instance it was recorded with — the
 * appKey MUST distinguish instances or a later generation would replay into an
 * earlier generation's (freed) KV buffers. This id is the driver's declaration
 * of that identity (part of "the bucket", §2.4 guard 2).
 */
const kvIds = new WeakMap<object, number>();
let nextKvId = 0;
export function staticKvId(kv: object): number {
  let id = kvIds.get(kv);
  if (id === undefined) {
    id = nextKvId++;
    kvIds.set(kv, id);
  }
  return id;
}

export function buildDecodeUploads(
  cfg: Cfg,
  posOffset: number,
  tokenId: number,
  bucketLen: number,
): Array<{ shape: number[]; values: Float32Array }> {
  const half = cfg.headDim / 2;
  const mask = new Float32Array(bucketLen).fill(-1e9);
  mask.fill(0, 0, posOffset + 1);
  return [
    { shape: [1, 1], values: Float32Array.from([tokenId]) },
    { shape: [1, half], values: new Float32Array(half).fill(posOffset) },
    {
      shape: [1, cfg.numKVHeads, 1, cfg.headDim],
      values: new Float32Array(cfg.numKVHeads * cfg.headDim).fill(posOffset),
    },
    { shape: [1, 1, 1, bucketLen], values: mask },
  ];
}
