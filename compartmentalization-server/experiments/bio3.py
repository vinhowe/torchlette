"""
BIO3 — factual compartmentalization on a synthetic "bio" task.

Entities have attributes with values. Multiple "compartments" encode the
same facts using disjoint token sets. The transformer has to decide
whether to learn separate fact stores per compartment or a unified
representation. Does cross-compartment supervision (translation pairs)
cause representations to merge?

Faithful port of examples/toy-compartmentalization/src/lib/bio-data.ts
and src/routes/bio/+page.svelte. Every knob the JS page exposes is
available as a script param here, and every metric the JS page plots
(loss, per-compartment QA accuracy, translation loss, cross-compartment
cosine similarity) is reported back to the coordinator.

Architecture: standard multi-head transformer (no attention bottleneck),
defaults to 2 layers, 4 heads, 64-dim, 256 MLP, learned positional
encoding. Differs from MESS3 in every architectural dimension.

Metrics:

  loss              cross-entropy on bio + QA + translation tokens
  acc_cN            every EVAL_INTERVAL — QA accuracy on attribute 0 for
                    compartment N (fraction of entities where the argmax
                    predicted value matches ground truth)
  cos_sim           middle-layer residual cosine similarity at the final
                    prompt position, compartment 0 vs 1, averaged over
                    the eval entity set. High = unified, low = split.
  translation_loss  NLL on the partB target tokens of mirror-mode
                    translation examples, averaged across all directed
                    compartment pairs. Measures whether the model can
                    reproduce a fact stated in compartment A using
                    compartment B's vocabulary.

Checkpoint semantics: model + optimizer + world RNG state + step. The
BioWorld itself is reconstructed from params on setup() and is
deterministic given the seed, so we don't need to serialize it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from compartmentalization_server.api import Experiment, register
from compartmentalization_server.models import (
    Transformer,
    TransformerConfig,
    hessian_lambda_max,
)

EVAL_INTERVAL = 50
# Sharpness is ~45 training-step-equivalents per call (15 power
# iterations × 3 passes each). Run it much less often than the cheap
# metrics to keep total overhead at ~0.1%.
SHARPNESS_INTERVAL = 500


# ──────────────────────────────────────────────────────────────────────────
# BioWorld
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class BioConfig:
    """All the knobs that shape the bio data distribution."""

    n_entities: int = 100
    n_attributes: int = 6
    n_values: int = 20
    n_compartments: int = 2
    tokens_per_entity: int = 1
    tokens_per_value: int = 1
    bank_size: int = 200  # base token bank per compartment (for multi-token names)
    mix_compartments: bool = False  # per-tuple compartment resampling inside a bio


@dataclass
class BioWorld:
    """A deterministic instantiation of the bio task: token layout + facts.

    Constructed from (BioConfig, seed) so two workers with the same
    config + seed produce identical training data. The RNG used for
    BATCH sampling is separate from the world-construction RNG so you
    can reseed mid-run without changing the fact table or token
    assignments.

    Token layout:
      [comp 0 bank] [comp 0 attr templates] [comp 0 qa prompts]
      [comp 1 bank] [comp 1 attr templates] [comp 1 qa prompts]
      ...
      [TR] [SEP]
    """

    config: BioConfig
    vocab_size: int
    # facts[entity_id][attribute_id] = value_id
    facts: list[list[int]]
    # entity_tokens[comp][entity_id] = list[int] (multi-token names)
    entity_tokens: list[list[list[int]]]
    # attr_tokens[comp][attr_id] = single int (one template token per attribute)
    attr_tokens: list[list[int]]
    # value_tokens[comp][value_id] = list[int]
    value_tokens: list[list[list[int]]]
    # qa_tokens[comp][attr_id] = single int (one prompt token per attribute)
    qa_tokens: list[list[int]]
    sep_token: int
    tr_token: int
    # Batch-time RNG (separate from world construction). A python stdlib
    # Random so we can reseed it deterministically without affecting the
    # rest of the process.
    batch_rng: random.Random = field(default_factory=random.Random)


def _assign_single_tokens(
    comp: int, count: int, offset: int, tokens_per_comp: int
) -> list[list[int]]:
    base = comp * tokens_per_comp
    return [[base + offset + i] for i in range(count)]


def _sample_unique_token_seq(
    comp: int,
    length: int,
    bank_size: int,
    tokens_per_comp: int,
    rng: random.Random,
    used: set[tuple[int, ...]],
    max_attempts: int = 1000,
) -> list[int]:
    base = comp * tokens_per_comp
    for _ in range(max_attempts):
        seq = tuple(base + rng.randrange(bank_size) for _ in range(length))
        if seq not in used:
            used.add(seq)
            return list(seq)
    raise RuntimeError(
        f"could not find a unique {length}-token sequence in comp {comp} "
        f"after {max_attempts} attempts — bank_size too small?"
    )


def create_bio_world(config: BioConfig, seed: int = 42) -> BioWorld:
    """Build a fresh BioWorld (token layout + ground-truth facts)."""
    if config.tokens_per_entity == 1 and config.bank_size < config.n_entities:
        raise ValueError(
            f"bank_size ({config.bank_size}) must be >= n_entities "
            f"({config.n_entities}) when tokens_per_entity=1"
        )
    if config.tokens_per_value == 1 and config.bank_size < config.n_values:
        raise ValueError(
            f"bank_size ({config.bank_size}) must be >= n_values "
            f"({config.n_values}) when tokens_per_value=1"
        )

    rng = random.Random(seed)

    # Layout:  [comp 0 bank][comp 0 attr templates][comp 0 qa prompts] | ... | [TR][SEP]
    n_comp = config.n_compartments
    special_per_comp = config.n_attributes + config.n_attributes  # attrs + qa prompts
    tokens_per_comp = config.bank_size + special_per_comp
    shared_base = n_comp * tokens_per_comp
    tr_token = shared_base
    sep_token = shared_base + 1
    vocab_size = shared_base + 2

    entity_tokens: list[list[list[int]]] = []
    value_tokens: list[list[list[int]]] = []
    attr_tokens: list[list[int]] = []
    qa_tokens: list[list[int]] = []

    for c in range(n_comp):
        spec_base = c * tokens_per_comp + config.bank_size

        at = [spec_base + a for a in range(config.n_attributes)]
        attr_tokens.append(at)
        qa = [spec_base + config.n_attributes + a for a in range(config.n_attributes)]
        qa_tokens.append(qa)

        if config.tokens_per_entity == 1:
            entity_tokens.append(
                _assign_single_tokens(c, config.n_entities, 0, tokens_per_comp)
            )
        else:
            used: set[tuple[int, ...]] = set()
            entity_tokens.append(
                [
                    _sample_unique_token_seq(
                        c, config.tokens_per_entity, config.bank_size,
                        tokens_per_comp, rng, used,
                    )
                    for _ in range(config.n_entities)
                ]
            )

        if config.tokens_per_value == 1:
            # Offset past entity tokens to avoid collision when both are single-token.
            val_offset = config.n_entities if config.tokens_per_entity == 1 else 0
            value_tokens.append(
                _assign_single_tokens(c, config.n_values, val_offset, tokens_per_comp)
            )
        else:
            used_v: set[tuple[int, ...]] = set()
            value_tokens.append(
                [
                    _sample_unique_token_seq(
                        c, config.tokens_per_value, config.bank_size,
                        tokens_per_comp, rng, used_v,
                    )
                    for _ in range(config.n_values)
                ]
            )

    # Ground-truth facts: facts[entity][attr] = value_id. Drawn from the
    # same world-RNG so two runs with the same seed share fact tables.
    facts: list[list[int]] = [
        [rng.randrange(config.n_values) for _ in range(config.n_attributes)]
        for _ in range(config.n_entities)
    ]

    return BioWorld(
        config=config,
        vocab_size=vocab_size,
        facts=facts,
        entity_tokens=entity_tokens,
        attr_tokens=attr_tokens,
        value_tokens=value_tokens,
        qa_tokens=qa_tokens,
        sep_token=sep_token,
        tr_token=tr_token,
        batch_rng=random.Random(seed),
    )


# ──────────────────────────────────────────────────────────────────────────
# Batch generation
# ──────────────────────────────────────────────────────────────────────────


def _generate_bio(world: BioWorld, entity_id: int, comp: int) -> list[int]:
    """One bio paragraph for (entity, comp). List of tokens, no padding."""
    cfg = world.config
    order = list(range(cfg.n_attributes))
    world.batch_rng.shuffle(order)
    mix = cfg.mix_compartments and cfg.n_compartments > 1

    seq: list[int] = []
    for attr_id in order:
        value_id = world.facts[entity_id][attr_id]
        # Each tuple is compartment-coherent; when mix is on, each tuple
        # independently resamples its compartment.
        tc = world.batch_rng.randrange(cfg.n_compartments) if mix else comp
        seq.extend(world.entity_tokens[tc][entity_id])
        seq.append(world.attr_tokens[tc][attr_id])
        seq.extend(world.value_tokens[tc][value_id])
        seq.append(world.sep_token)
    return seq


def _generate_qa(
    world: BioWorld, entity_id: int, attr_id: int, comp: int
) -> tuple[list[int], list[int]]:
    """QA pair: (prompt tokens, target value tokens) for one (entity, attr, comp)."""
    value_id = world.facts[entity_id][attr_id]
    prompt = [*world.entity_tokens[comp][entity_id], world.qa_tokens[comp][attr_id]]
    target = list(world.value_tokens[comp][value_id])
    return prompt, target


def _generate_translation(
    world: BioWorld,
    entity_id: int,
    attr_id: int,
    comp_a: int,
    comp_b: int,
    mode: str = "mirror",
) -> list[int]:
    """One translation example between two compartments for a given fact."""
    cfg = world.config
    value_id = world.facts[entity_id][attr_id]
    part_a = [
        *world.entity_tokens[comp_a][entity_id],
        world.attr_tokens[comp_a][attr_id],
        *world.value_tokens[comp_a][value_id],
    ]
    part_b = [
        *world.entity_tokens[comp_b][entity_id],
        world.attr_tokens[comp_b][attr_id],
        *world.value_tokens[comp_b][value_id],
    ]

    if mode == "mirror":
        return [world.tr_token, *part_a, world.tr_token, *part_b]
    if mode == "continuation":
        split_idx = 1 + world.batch_rng.randrange(max(1, len(part_a) - 1))
        return [*part_a[:split_idx], world.tr_token, *part_b[split_idx:]]
    # dictionary: token-pair only, no fact context
    pair_kind = world.batch_rng.randrange(3)
    if pair_kind == 0:
        return [
            world.tr_token,
            *world.entity_tokens[comp_a][entity_id],
            *world.entity_tokens[comp_b][entity_id],
        ]
    if pair_kind == 1:
        return [
            world.tr_token,
            world.attr_tokens[comp_a][attr_id],
            world.attr_tokens[comp_b][attr_id],
        ]
    return [
        world.tr_token,
        *world.value_tokens[comp_a][value_id],
        *world.value_tokens[comp_b][value_id],
    ]


def generate_training_batch(
    world: BioWorld,
    batch_size: int,
    seq_len: int,
    translation_frac: float,
    translation_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce (tokens [B, T] int64, targets [B, T] int64).

    Targets of -1 indicate positions that shouldn't contribute to the
    loss (padding, QA prompt positions, etc.). Matches the JS
    generateTrainingBatch mix:

      bio   = 0.7 * (1 − t)
      qa    = 0.3 * (1 − t)
      trans = t
    """
    cfg = world.config
    t_frac = max(0.0, min(1.0, translation_frac))
    bio_threshold = 0.7 * (1.0 - t_frac)
    qa_threshold = 1.0 - t_frac

    tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
    targets = torch.full((batch_size, seq_len), -1, dtype=torch.long)

    for b in range(batch_size):
        comp = world.batch_rng.randrange(cfg.n_compartments)
        r = world.batch_rng.random()

        if r < bio_threshold:
            entity_id = world.batch_rng.randrange(cfg.n_entities)
            seq = _generate_bio(world, entity_id, comp)
            tgt = [-1] * len(seq)
            for i in range(len(seq) - 1):
                tgt[i] = seq[i + 1]
        elif r < qa_threshold:
            entity_id = world.batch_rng.randrange(cfg.n_entities)
            attr_id = world.batch_rng.randrange(cfg.n_attributes)
            prompt, target = _generate_qa(world, entity_id, attr_id, comp)
            seq = [*prompt, *target]
            tgt = [-1] * len(seq)
            # Loss only on value tokens (after the prompt)
            for i in range(len(prompt), len(seq)):
                tgt[i - 1] = seq[i]
        else:
            entity_id = world.batch_rng.randrange(cfg.n_entities)
            attr_id = world.batch_rng.randrange(cfg.n_attributes)
            if cfg.n_compartments > 1:
                offset = 1 + world.batch_rng.randrange(cfg.n_compartments - 1)
                comp_b = (comp + offset) % cfg.n_compartments
            else:
                comp_b = comp
            seq = _generate_translation(
                world, entity_id, attr_id, comp, comp_b, translation_mode
            )
            tgt = [-1] * len(seq)
            for i in range(len(seq) - 1):
                tgt[i] = seq[i + 1]

        n = min(len(seq), seq_len)
        tokens[b, :n] = torch.tensor(seq[:n], dtype=torch.long)
        m = min(len(tgt), seq_len)
        targets[b, :m] = torch.tensor(tgt[:m], dtype=torch.long)

    return tokens, targets


def generate_eval_qa(
    world: BioWorld, entity_ids: list[int], attr_id: int, seq_len: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int, int]:
    """Build per-compartment QA eval batches.

    Returns (tokens[c] [n, seq_len], targets[c] [n, target_len], eval_seq_len,
    prompt_len, target_len). eval_seq_len is the length of the QA example
    BEFORE padding to seq_len; prompt_len and target_len are the
    compartment-agnostic shape (they're identical for all comps because
    the token banks have the same size per comp).
    """
    cfg = world.config
    probe_prompt, probe_target = _generate_qa(world, entity_ids[0], attr_id, 0)
    prompt_len = len(probe_prompt)
    target_len = len(probe_target)
    eval_seq_len = prompt_len + target_len

    tokens_per_comp: list[torch.Tensor] = []
    targets_per_comp: list[torch.Tensor] = []
    for c in range(cfg.n_compartments):
        tok = torch.zeros((len(entity_ids), seq_len), dtype=torch.long)
        tgt = torch.zeros((len(entity_ids), target_len), dtype=torch.long)
        for i, eid in enumerate(entity_ids):
            prompt, target = _generate_qa(world, eid, attr_id, c)
            for t, v in enumerate(prompt):
                tok[i, t] = v
            for t, v in enumerate(target):
                tok[i, prompt_len + t] = v
                tgt[i, t] = v
        tokens_per_comp.append(tok)
        targets_per_comp.append(tgt)

    return tokens_per_comp, targets_per_comp, eval_seq_len, prompt_len, target_len


def generate_eval_translation(
    world: BioWorld, entity_ids: list[int], attr_id: int, seq_len: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]], int, int, int]:
    """Mirror-mode translation eval batch for every directed comp pair.

    For each (a, b) with a != b, builds one example per entity:
      [TR, partA, TR, partB]
    Returns (tokens_per_pair, targets_per_pair, pairs, translation_seq_len,
    prompt_len, target_len). The "targets" are the partB tokens aligned so
    the caller can compute CE over the target positions only.
    """
    cfg = world.config
    if cfg.n_compartments < 2:
        return [], [], [], 0, 0, 0

    probe = _generate_translation(world, entity_ids[0], attr_id, 0, 1, "mirror")
    trans_seq_len = len(probe)
    part_a_len = cfg.tokens_per_entity + 1 + cfg.tokens_per_value
    prompt_len = 1 + part_a_len + 1  # TR + partA + TR
    target_len = trans_seq_len - prompt_len

    tokens_per_pair: list[torch.Tensor] = []
    targets_per_pair: list[torch.Tensor] = []
    pairs: list[tuple[int, int]] = []
    for a in range(cfg.n_compartments):
        for b in range(cfg.n_compartments):
            if a == b:
                continue
            tok = torch.zeros((len(entity_ids), seq_len), dtype=torch.long)
            tgt = torch.zeros((len(entity_ids), target_len), dtype=torch.long)
            for i, eid in enumerate(entity_ids):
                seq = _generate_translation(world, eid, attr_id, a, b, "mirror")
                for t in range(trans_seq_len):
                    tok[i, t] = seq[t]
                for t in range(target_len):
                    tgt[i, t] = seq[prompt_len + t]
            tokens_per_pair.append(tok)
            targets_per_pair.append(tgt)
            pairs.append((a, b))
    return tokens_per_pair, targets_per_pair, pairs, trans_seq_len, prompt_len, target_len


# ──────────────────────────────────────────────────────────────────────────
# Experiment
# ──────────────────────────────────────────────────────────────────────────


@register(
    "bio3",
    description="Bio3: factual compartmentalization on a synthetic entity/attribute/value task",
    params={
        # ── world ──
        "n_entities": {
            "type": "number", "default": 100, "min": 10, "max": 1000, "scale": "linear",
            "live": False, "description": "Number of distinct entities",
        },
        "n_attributes": {
            "type": "number", "default": 6, "min": 1, "max": 20, "scale": "linear",
            "live": False, "description": "Attributes per entity",
        },
        "n_values": {
            "type": "number", "default": 20, "min": 5, "max": 100, "scale": "linear",
            "live": False, "description": "Possible values per attribute",
        },
        "n_compartments": {
            "type": "number", "default": 2, "min": 1, "max": 8, "scale": "linear",
            "live": False, "description": "Parallel surface-form encodings of the same facts",
        },
        "tokens_per_entity": {
            "type": "number", "default": 1, "min": 1, "max": 3, "scale": "linear",
            "live": False, "description": "Tokens per entity name (1=single, 2-3=compound)",
        },
        "tokens_per_value": {
            "type": "number", "default": 1, "min": 1, "max": 2, "scale": "linear",
            "live": False, "description": "Tokens per value",
        },
        "mix_compartments": {
            "type": "boolean", "default": False, "live": False,
            "description": "Each bio-paragraph tuple resamples its compartment independently",
        },
        # ── objective ──
        "translation_pct": {
            "type": "number", "default": 0, "min": 0, "max": 100, "scale": "linear",
            "live": True,
            "description": "% of the training mix that's translation pairs (cross-compartment supervision)",
        },
        "translation_mode": {
            "type": "select", "choices": ["mirror", "continuation", "dictionary"],
            "default": "mirror", "live": True,
            "description": "How translation examples are formatted",
        },
        # ── init ──
        "seed": {
            "type": "number", "default": 1024, "min": 0, "max": 100000, "scale": "linear",
            "live": False, "description": "RNG seed for world construction + init",
        },
        "tie_init": {
            "type": "boolean", "default": False, "live": False,
            "description": "Initialize all compartments' embeddings identically",
        },
        "weight_scale": {
            "type": "number", "default": 1.0, "min": 0.1, "max": 10.0, "scale": "log",
            "live": False, "description": "Multiplicative scale on all parameter tensors at init",
        },
        "embed_scale": {
            "type": "number", "default": 1.0, "min": 0.0, "max": 2.0, "scale": "linear",
            "live": False, "description": "Multiplicative scale on wte at init",
        },
        "head_scale": {
            "type": "number", "default": 1.0, "min": 0.0, "max": 2.0, "scale": "linear",
            "live": False, "description": "Multiplicative scale on lm_head at init",
        },
        "residual_zero_init": {
            "type": "boolean", "default": False, "live": False,
            "description": "Zero the attn out_proj and fc2 weights at init (prevents residual stream drift)",
        },
        # ── model ──
        "embed_dim": {
            "type": "number", "default": 64, "min": 16, "max": 512, "scale": "linear",
            "live": False, "description": "Residual stream width",
        },
        "num_layers": {
            "type": "number", "default": 2, "min": 1, "max": 12, "scale": "linear",
            "live": False, "description": "Number of transformer blocks",
        },
        "num_heads": {
            "type": "number", "default": 4, "min": 1, "max": 16, "scale": "linear",
            "live": False, "description": "Attention heads",
        },
        "seq_len": {
            "type": "number", "default": 64, "min": 16, "max": 256, "scale": "linear",
            "live": False, "description": "Context length",
        },
        "pos_encoding": {
            "type": "select", "choices": ["learned", "rope"], "default": "learned",
            "live": False, "description": "Positional encoding",
        },
        # ── optim ──
        "lr": {
            "type": "number", "default": 1e-4, "min": 1e-5, "max": 1e-1, "scale": "log",
            "live": True, "description": "Adam learning rate",
        },
        "weight_decay": {
            "type": "number", "default": 0.0, "min": 0.0, "max": 0.2, "scale": "linear",
            "live": True, "description": "Adam weight decay (AdamW when > 0)",
        },
        "batch_size": {
            "type": "number", "default": 32, "min": 8, "max": 128, "scale": "linear",
            "live": True, "description": "Sequences per training step",
        },
    },
)
class Bio3(Experiment):
    def setup(self) -> None:
        # Build the world first so vocab_size is known before the model.
        bio_cfg = BioConfig(
            n_entities=int(self.params["n_entities"]),
            n_attributes=int(self.params["n_attributes"]),
            n_values=int(self.params["n_values"]),
            n_compartments=int(self.params["n_compartments"]),
            tokens_per_entity=int(self.params["tokens_per_entity"]),
            tokens_per_value=int(self.params["tokens_per_value"]),
            bank_size=max(
                200,
                int(self.params["n_entities"]) + int(self.params["n_values"]) + 10,
            ),
            mix_compartments=bool(self.params["mix_compartments"]),
        )
        self.world = create_bio_world(bio_cfg, seed=int(self.params["seed"]))
        self.bio_cfg = bio_cfg

        self.seq_len = int(self.params["seq_len"])
        self.eval_entity_ids = list(range(bio_cfg.n_entities))

        tf_cfg = TransformerConfig(
            vocab_size=self.world.vocab_size,
            seq_len=self.seq_len,
            embed_dim=int(self.params["embed_dim"]),
            num_heads=int(self.params["num_heads"]),
            num_layers=int(self.params["num_layers"]),
            mlp_dim=int(self.params["embed_dim"]) * 4,
            pos_encoding=str(self.params["pos_encoding"]),
        )
        self.model = Transformer(tf_cfg).to(self.device)
        self._apply_init_scaling()

        weight_decay = float(self.params["weight_decay"])
        optim_cls = torch.optim.AdamW if weight_decay > 0 else torch.optim.Adam
        self.optimizer = optim_cls(
            self.model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=weight_decay,
        )

    def _apply_init_scaling(self) -> None:
        """Apply the init-time multiplicative scales + zero-init knobs.

        This runs after the default nn.init for each parameter, so it's
        purely a post-init fix-up. `tie_init` copies compartment 0's
        token embeddings into every other compartment's slots so they
        start from the same representation.
        """
        ws = float(self.params["weight_scale"])
        es = float(self.params["embed_scale"])
        hs = float(self.params["head_scale"])
        rz = bool(self.params["residual_zero_init"])
        tie = bool(self.params["tie_init"])

        with torch.no_grad():
            if ws != 1.0:
                for p in self.model.parameters():
                    p.mul_(ws)
            if es != 1.0:
                self.model.wte.weight.mul_(es)
            if hs != 1.0:
                self.model.lm_head.weight.mul_(hs)
            if rz:
                for block in self.model.blocks:
                    block.attn.out_proj.weight.zero_()
                    block.fc2.weight.zero_()
            if tie and self.world.config.n_compartments > 1:
                self._tie_compartment_embeddings()

    def _tie_compartment_embeddings(self) -> None:
        """Copy compartment 0's rows of wte+lm_head into every other comp.

        Runs under no_grad (caller holds the context). Doesn't do anything
        clever about per-token frequency — just overwrites.
        """
        wte = self.model.wte.weight
        lm = self.model.lm_head.weight
        cfg = self.world.config

        def copy_row(src: int, dst: int) -> None:
            wte[dst].copy_(wte[src])
            lm[dst].copy_(lm[src])

        for c in range(1, cfg.n_compartments):
            for eid in range(cfg.n_entities):
                for t in range(cfg.tokens_per_entity):
                    copy_row(
                        self.world.entity_tokens[0][eid][t],
                        self.world.entity_tokens[c][eid][t],
                    )
            for a in range(cfg.n_attributes):
                copy_row(self.world.attr_tokens[0][a], self.world.attr_tokens[c][a])
                copy_row(self.world.qa_tokens[0][a], self.world.qa_tokens[c][a])
            for v in range(cfg.n_values):
                for t in range(cfg.tokens_per_value):
                    copy_row(
                        self.world.value_tokens[0][v][t],
                        self.world.value_tokens[c][v][t],
                    )

    def step(self) -> dict[str, float]:
        # Live params
        lr = float(self.params["lr"])
        batch_size = int(self.params["batch_size"])
        wd = float(self.params["weight_decay"])
        trans_pct = float(self.params["translation_pct"])
        trans_mode = str(self.params["translation_mode"])
        for g in self.optimizer.param_groups:
            g["lr"] = lr
            g["weight_decay"] = wd

        tokens, targets = generate_training_batch(
            self.world,
            batch_size=batch_size,
            seq_len=self.seq_len,
            translation_frac=trans_pct / 100.0,
            translation_mode=trans_mode,
        )
        tokens = tokens.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # Next-token CE: predict tokens[:, 1:] from positions [:, :-1],
        # ignoring positions where the batch generator set target=-1.
        inputs = tokens[:, :-1]
        shifted_targets = targets[:, :-1]  # targets already holds shifted values
        logits = self.model(inputs)  # [B, T-1, V]
        loss = F.cross_entropy(
            logits.reshape(-1, self.world.vocab_size),
            shifted_targets.reshape(-1),
            ignore_index=-1,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Matches the JS bio page's clipGradNorm_(model.parameters(), 1.0).
        # The clip function returns the PRE-clip total L2 norm, which is
        # the sharpness metric the user asked for: spikes = cliff in the
        # loss landscape, steady decrease = well-behaved training.
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
        )
        self.optimizer.step()

        metrics: dict[str, float] = {
            "loss": float(loss.detach().item()),
            "grad_norm": grad_norm,
        }

        # Cheap eval: QA accuracy + cos_sim + translation loss.
        if (self.step_count + 1) % EVAL_INTERVAL == 0:
            metrics.update(self._eval())

        # Expensive eval: sharpness (Hessian λ_max). Own cadence to
        # cap overhead.
        if (self.step_count + 1) % SHARPNESS_INTERVAL == 0:
            sharpness = self._compute_sharpness()
            if sharpness is not None:
                metrics["sharpness"] = sharpness

        return metrics

    @torch.no_grad()
    def _eval(self) -> dict[str, float]:
        self.model.eval()
        try:
            return self._eval_inner()
        finally:
            self.model.train()

    def _eval_inner(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        cfg = self.world.config
        attr_id = 0

        # ── QA accuracy + cross-compartment cosine similarity ──
        tokens_per_comp, targets_per_comp, eval_sl, prompt_len, target_len = generate_eval_qa(
            self.world, self.eval_entity_ids, attr_id, self.seq_len
        )

        mid_residuals: list[torch.Tensor] = []  # one per compartment
        for c in range(cfg.n_compartments):
            tok = tokens_per_comp[c].to(self.device, non_blocking=True)
            tgt = targets_per_comp[c].to(self.device, non_blocking=True)  # [N, target_len]
            logits, residuals = self.model(tok, return_residuals=True)  # [N, SL, V]
            mid_residuals.append(residuals[len(residuals) // 2])

            # Accuracy: argmax at positions [prompt_len-1 .. prompt_len-1+target_len-1]
            # (predict target[i] from position prompt_len - 1 + i).
            n = tok.shape[0]
            correct = torch.ones(n, dtype=torch.bool, device=self.device)
            for t in range(target_len):
                pos = prompt_len - 1 + t
                pred = logits[:, pos, :].argmax(dim=-1)  # [N]
                correct &= pred == tgt[:, t]
            metrics[f"acc_c{c}"] = float(correct.float().mean().item())

        # Cosine similarity: middle residual at last prompt position,
        # compartment 0 vs compartment 1. With >2 compartments we still
        # only report 0↔1 for a single scalar metric — the JS page does
        # the same (it picks `Math.min(1, nCompartments-1)`).
        if cfg.n_compartments >= 2:
            cos_pos = prompt_len - 1
            a = mid_residuals[0][:, cos_pos, :]  # [N, D]
            b = mid_residuals[1][:, cos_pos, :]
            metrics["cos_sim"] = float(F.cosine_similarity(a, b, dim=-1).mean().item())

        # ── Translation loss (mirror mode, averaged over directed pairs) ──
        if cfg.n_compartments >= 2:
            trans_tokens, trans_targets, pairs, _, trans_prompt_len, trans_target_len = (
                generate_eval_translation(
                    self.world, self.eval_entity_ids, attr_id, self.seq_len
                )
            )
            total_nll = 0.0
            total_positions = 0
            for p_idx in range(len(pairs)):
                tok = trans_tokens[p_idx].to(self.device, non_blocking=True)
                tgt = trans_targets[p_idx].to(self.device, non_blocking=True)
                logits = self.model(tok)  # [N, SL, V]
                for t in range(trans_target_len):
                    pos = trans_prompt_len - 1 + t
                    step_logits = logits[:, pos, :]  # [N, V]
                    step_tgt = tgt[:, t]  # [N]
                    nll = F.cross_entropy(step_logits, step_tgt, reduction="sum")
                    total_nll += float(nll.item())
                    total_positions += step_tgt.shape[0]
            if total_positions > 0:
                metrics["translation_loss"] = total_nll / total_positions

        return metrics

    def _compute_sharpness(self) -> float | None:
        """Hessian λ_max on a fresh training-distribution minibatch.

        Expensive (~45 training-step-equivalents per call); gated on
        SHARPNESS_INTERVAL in step() so it runs much less often than
        the other eval metrics. Returns None on numerical failure
        rather than crashing the step.
        """
        sharpness_bs = 32

        def _sharpness_loss() -> torch.Tensor:
            tokens, targets = generate_training_batch(
                self.world,
                batch_size=sharpness_bs,
                seq_len=self.seq_len,
                translation_frac=float(self.params["translation_pct"]) / 100.0,
                translation_mode=str(self.params["translation_mode"]),
            )
            tokens = tokens.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            inputs = tokens[:, :-1]
            shifted_targets = targets[:, :-1]
            logits = self.model(inputs)
            return F.cross_entropy(
                logits.reshape(-1, self.world.vocab_size),
                shifted_targets.reshape(-1),
                ignore_index=-1,
            )

        params = [p for p in self.model.parameters() if p.requires_grad]
        try:
            return hessian_lambda_max(_sharpness_loss, params, num_iters=15)
        except Exception:
            return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # getstate/setstate pickle the python RNG's internal state so
            # batch draws resume deterministically. JSON-safe via
            # torch.save's pickler (this goes into checkpoint.pt, not
            # metadata.json).
            "batch_rng_state": self.world.batch_rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if "batch_rng_state" in state:
            self.world.batch_rng.setstate(state["batch_rng_state"])
        # See mess3.load_state_dict — optimizer state tensors need to be
        # moved back to the experiment's device after a CPU load.
        for opt_state in self.optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)
