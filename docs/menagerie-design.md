# Menagerie — a Picbreeder for models

> Working title. An open-ended, anarchic, genealogical ecology of models that
> visitors train, fork, merge, and evaluate — entirely in the browser, with
> Hugging Face as the durable store and identity layer.

This doc is the **foundation contract**: the HF repo layout, the provenance /
genealogy data model, and the community-validatable eval mechanism. Get this
right and everything else (avatars, merging, LoRA, DiLoCo) is additive.

## North star

- **Humans breed, objectives don't gate.** Novelty-over-objective (Picbreeder
  ethos). The population *is* the artifact. Evals are opt-in *lenses*, never
  gates.
- **No anonymity, no shared-write surface.** Every action is an HF commit under
  the actor's own identity, in the actor's own namespace. Spammers can only
  pollute their own corner.
- **Everything verifiable.** Public weights at a pinned commit SHA → every
  provenance claim and every eval score is independently reproducible, not a
  trusted self-report.

## Identity & auth

- HF OAuth 2.0 / OIDC via `@huggingface/hub` (`oauthLoginUrl`,
  `oauthHandleRedirectIfPresent`). Scopes: `read-repos`, `write-repos`,
  `profile`. Token stays client-side.
- Register a HF OAuth app; the redirect is **our own website**, not a HF Space.
- The token only writes to the logged-in user's namespace. Forks land in the
  forker's namespace. There is no shared mutable repo.

## What is "a model" here

A model is **one HF model repo**. Its git history is the model's life: each
durable checkpoint is a commit. Forks are *new repos* (cross-user) seeded from a
parent's snapshot at a pinned commit SHA. A model belongs to the ecology iff its
model card carries the `menagerie` tag (discovery is a Hub search for that tag).

> HF is a snapshot/release store, **not** a per-batch training-state store.
> Per-batch commits balloon LFS history (binary blobs aren't delta-compressed).
> Ephemeral per-batch churn (for live/DiLoCo later) lives off-HF on a relay; only
> durable milestones + forks are committed. MVP commits one snapshot per
> explicit "save" action.

### Repo layout (per model repo)

```
README.md            # model card: YAML frontmatter (tags incl. `menagerie`, lineage summary, avatar seed) + human notes
config.json          # arch preset id + dims (n_layer/n_head/n_embd/vocab/ctx), dtype, tokenizer id
model.safetensors    # weights at HEAD (LFS)
lineage.json         # provenance for THIS repo's life (see below)
diet/                # bundled private training text (becomes public under the uploader's identity)
  <sha>.txt
evals/               # append-only community eval results (see eval section)
  <eval-id>/
    <submitter-username>.json
adapters/            # (later) LoRA adapters not yet absorbed
  <name>.safetensors
```

### `lineage.json`

One file per repo, append-only `checkpoints` array. The cross-repo genealogy DAG
is reconstructed by following `parent` pointers across repos (a Hub tag search
gives the node set; `parent` edges give the graph; merge nodes have two parents).

```jsonc
{
  "schema": 1,
  "root_repo": "alice/menagerie-aardvark",         // ultimate ancestor
  "parent": {                                        // null for from-scratch roots
    "repo": "bob/menagerie-okapi",
    "commit_sha": "e9f7943..."                       // pinned snapshot we forked
  },
  "arch_preset": "nano-1m",                          // tier id (see presets)
  "checkpoints": [
    {
      "commit_sha": "<filled by us post-commit>",
      "op": "fork" | "train" | "merge" | "lora-absorb" | "init",
      "created_by": "alice",                         // HF username (renameable; display + namespace)
      "created_by_sub": "62f0…",                     // stable HF user id — identity anchor (from profile)
      "created_at": "2026-06-19T...Z",
      "tokens": 1048576,
      "steps": 200,
      "wallclock_ms": 412000,
      "dtype": "f16",
      "hparams": { "lr": 3e-4, "batch": 8, "seq": 512, "optimizer": "adamw", "wd": 0.1 },
      "client": { "ua": "...", "webgpu_adapter": "...", "tier_recommended": "laptop", "tier_used": "laptop" },
      "diet": [ /* dataset diet entries, see below */ ],
      "merge_parents": [ /* for op=merge: [{repo, commit_sha, weight}] */ ]
    }
  ]
}
```

### Dataset diet provenance

Two kinds, both recorded per checkpoint in `diet[]`:

- **Public dataset rows** — stored as a *reference*, resolved live for display:
  ```jsonc
  { "kind": "hf-dataset", "dataset": "HuggingFaceFW/fineweb-edu", "config": "default",
    "split": "train", "rows": { "offset": 0, "length": 50000 },
    "fingerprint": "sha256:..." }   // hash of the resolved content, so the slice is pinned
  ```
  Displayed by hitting `datasets-server.huggingface.co/rows` (same path the
  existing pretrain-v2 trainer already uses).
- **Uploaded private text** — *bundled* into `diet/<sha>.txt` at that checkpoint.
  Becomes public under the uploader's HF identity. Recorded as
  `{ "kind": "bundled", "path": "diet/<sha>.txt", "fingerprint": "sha256:...", "tokens": N }`.

The fingerprints double as the **contamination check**: an eval can flag a model
whose diet fingerprints / n-gram overlap intersect the eval's own row set
(train-on-val), shown as a caveat on the eval badge.

## Architecture presets (choose-your-own-adventure by hardware)

Fixed set of presets, each a `(n_layer, n_head, n_embd, ctx, vocab)` + dtype.
The smallest must run on a typical iPhone (below distilgpt2). WebGPU does **not**
expose total VRAM — we recommend a tier from `adapter.limits`
(`maxBufferSize`, `maxStorageBufferBindingSize`) + a probe allocation, let the
user override, and downshift on OOM.

| tier id     | rough target            | params | notes                              |
|-------------|-------------------------|--------|------------------------------------|
| `nano-1m`   | iPhone / mobile Safari  | ~1M    | tight storage-buffer limits        |
| `micro-10m` | low-end laptop          | ~10M   |                                    |
| `mini-distil` | typical laptop        | ~82M   | ≈ distilgpt2                       |
| `base-124m` | desktop / good GPU      | ~124M  | the DiLoCo regression size         |
| `big-*`     | "only a few can train"  | 350M+  | introduced by capable clients      |

A fork may start **from scratch** (random init at the chosen preset) or **from
an existing HF checkpoint** (seed weights, possibly a different preset → resize
later; MVP keeps preset fixed across a fork).

## Genealogy operations (the verbs)

- **fork** — `createRepo` in actor's namespace; seed `model.safetensors` +
  `config.json` from parent @ pinned SHA; write `lineage.json` with `parent`.
- **train** — load HEAD weights, train N steps in-browser, commit new snapshot +
  append a `train` checkpoint.
- **merge** *(on-device, later in MVP)* — elementwise weighted mean of aligned
  tensors; **gated by mode-connectivity** (common-ancestor-within-N-forks check
  on the DAG). Two-parent node.
- **lora-absorb** *(heavy tier, later)* — fold adapter stack `W ← W + ΣBᵢAᵢ` into
  a clean base; a genealogy node. Enables the "forage (small devices add LoRAs) →
  consolidate (big rigs absorb)" ecology.

## Avatars (Picbreeder/Gravatar-style)

A **separate tiny CPPN painter**, genome = deterministic function of the model
identity hash (root_repo + lineage). **Zero direct user control** (abuse
mitigation). Inherits along genealogy with mutation; drift ∝ weight-delta, so the
avatar is a visual grammar for the verbs (fork = mutate, train-a-lot = bigger
mutation, merge = CPPN crossover, LoRA = small perturbation). Decoupled from the
model's own weights so it's always a pretty abstract image.

## Community-validatable evals  ← up-front requirement

**Goal:** 0+ people can add an eval result to any model; a *validated* badge marks
scores that independent parties re-ran on the exact weights and agree on. This
keeps high-eval-score claims honest without a central authority.

### Standardized eval registry

Evals are *standardized definitions* so everyone runs the same thing. MVP ships a
small static registry (in-app, versioned); later it can be community-extended.

```jsonc
// eval definition (in-app registry, versioned)
{ "id": "tinystories-ppl@1",
  "name": "TinyStories perplexity",
  "kind": "perplexity",                    // perplexity | accuracy | ...
  "dataset_ref": { "dataset": "roneneldan/TinyStories", "config": "default",
                   "split": "validation", "rows": { "offset": 0, "length": 2000 },
                   "fingerprint": "sha256:..." },
  "metric": "nats_per_token",              // lower better; direction declared
  "direction": "lower",
  "harness_version": "1",                  // bump invalidates prior validations
  "deterministic": true }                  // greedy / fixed-order → reproducible
```

The eval **runs in-browser** on the pinned weights (same torchlette forward path
the trainer uses), over a pinned, fingerprinted dataset slice. Determinism is the
whole point — same weights + same eval id + same harness version ⇒ same score
(to a declared tolerance).

### Eval result (one file per submitter, append-only)

Stored in the **evaluated model's** repo, committed by the submitter under their
own identity (or, if they lack write access to someone else's repo — they will,
cross-user — submitted to a sibling "evals" channel; see open question below).

```jsonc
// evals/tinystories-ppl@1/carol.json
{ "eval_id": "tinystories-ppl@1",
  "harness_version": "1",
  "commit_sha": "<exact weights evaluated>",   // pins WHICH weights
  "submitter": "carol",                        // HF username (display)
  "submitter_sub": "8a3f…",                    // stable HF user id — the distinctness key for validation
  "submitted_at": "2026-06-19T...Z",
  "score": 3.214,
  "raw": { "total_nll": ..., "tokens": ... },  // enough to recompute the score
  "client": { "webgpu_adapter": "...", "ua": "..." },
  "contamination": { "diet_overlap_rows": 0, "ngram_overlap": 0.0 } }  // self-reported, recomputable
```

### Validated badge (derived, not stored as trust)

A score for `(commit_sha, eval_id, harness_version)` is **validated** when ≥2
results from **distinct HF submitters** agree within the eval's tolerance. The
badge is *computed* from the set of result files at view time — never a flag
someone can set. UI states:

- **claimed** — 1 submitter (likely the model owner). Show the number, no badge.
- **validated ✓** — ≥2 distinct submitters agree (within tolerance).
- **disputed ⚠** — distinct submitters disagree beyond tolerance (show spread).
- **stale** — results exist only for an older `harness_version`.

Because weights are public at a pinned SHA, *anyone* can click "re-run this eval"
to add their own corroborating result — that is the entire integrity mechanism.

### Cross-user eval submission — DECIDED: per-user eval repos, aggregated by scan

Carol can't `write-repos` to Bob's repo, so her eval result for Bob's model
lands in **her own namespace**, and the app aggregates by scanning. Concretely:

**Each submitter owns ONE HF _dataset_ repo**, e.g. `carol/menagerie-evals`,
tagged `menagerie-evals`. Inside it:

```
index.json                                  # lightweight, scannable summary (append-only)
results/<target-owner>__<target-name>/<commit_sha>/<eval_id>.json   # full result
```

`index.json` is a flat array of the minimum needed to compute badges without
opening every result file:

```jsonc
[ { "target_repo": "bob/foo", "commit_sha": "abc123…", "eval_id": "tinystories-ppl@1",
    "harness_version": "1", "score": 3.21, "submitted_at": "2026-06-19T…Z" } ]
```

**Material aggregation** — how the app turns scattered per-user files into a
badge on `bob/foo` (no central writable index, no owner action):

1. `GET /api/datasets?filter=menagerie-evals` → the set of submitter eval repos
   (one request; returns repo ids + `lastModified`).
2. For each repo, fetch its `index.json` (N parallel fetches, **N = number of
   people who've ever submitted any eval**, *not* number of results). Cache
   keyed on repo + `lastModified` so repeat views are nearly free.
3. Flatten all entries, filter `target_repo === "bob/foo"`, group by
   `(commit_sha, eval_id, harness_version)`.
4. Per group → distinct submitters **keyed on the stable HF user id (`sub`),
   not username** (usernames are renameable / reusable): ≥2 agree within
   tolerance ⇒ **validated**;
   disagree ⇒ **disputed**; only an older harness ⇒ **stale**; exactly 1 ⇒
   **claimed**.
5. The full `results/.../<eval_id>.json` (raw counts, client, contamination) is
   fetched lazily only when a user expands that specific eval.

**Cost / scale.** O(distinct submitters) small cached fetches, fully
parallelizable — fine for hundreds of submitters. If it ever outgrows that, a
single aggregator dataset rebuilt by a Space/cron doing exactly this scan
becomes the index and the client reads one file; MVP does not need it.

**Trust model.** A submitter *can* write a false score in their own
`index.json` — which is precisely why a lone entry is only **claimed**, never
validated. Validation requires ≥2 *independent* HF identities agreeing, and
because the weights are public at the pinned SHA, anyone can re-run to confirm
or dispute. Residual attack = Sybil (many accounts self-validating); out of
scope for MVP, mitigable later by weighting on account age/activity.

## MVP phasing (each phase ends green + committed)

0. **Scaffold** — new SvelteKit app reusing torchlette + the existing
   browser-trainer modules. This doc. ← start here
1. **Auth + discovery** — Login with HF; list ecology models (Hub tag search);
   model detail page reading `config.json`/`lineage.json`.
2. **Fork** — createRepo + seed-from-parent-SHA + write lineage.
3. **Train + snapshot** — in-browser train (reuse pretrain-v2 loop) → commit
   weights + append checkpoint + diet provenance.
4. **Evals + badges** — in-app eval registry; run eval in-browser; submit result
   (per decision above); derived validated/disputed/stale badges.
5. **Genealogy view + CPPN avatar** — DAG visualization; deterministic avatars.
6. **Later** — on-device merge (mode-connectivity gated), first-class LoRA +
   absorption, DiLoCo / real-time passing (the pièce de résistance),
   embedding-similarity-search-by-dataset, distillation (heavy tier),
   architecture-changing forks via a restricted DSL.

## Deferred / explicitly out of MVP

- Arbitrary-JS forks with sandboxing (only if ever needed).
- Architecture-changing forks (restricted DSL) — validate the basic premise
  first.
- Heritable names as anything more than a decorative overlay (genus + species +
  hash suffix), not identity.
