#!/usr/bin/env python3
"""
Summarize STATS JSONL lines emitted by tools/diloco-webrtc-agent.ts.

Reads from a file argument or stdin. Lines not starting with "STATS " are
ignored, so you can pipe raw agent stderr through.

Usage:
  tools/analyze-stats.py /tmp/agent-1.log
  tail -F /tmp/agent-1.log | tools/analyze-stats.py
  tools/analyze-stats.py /tmp/agent-1.log /tmp/agent-2.log
  WARMUP_ROUNDS=5 tools/analyze-stats.py /tmp/agent-1.log

Output: a compact text summary with steady-state values, trend slopes (via
least-squares on round index), and warning flags. Designed to be eyeballable
during a long run and useful for post-mortem.

Slope computation skips the first WARMUP_ROUNDS rounds (default 3) — JS heap
warmup, lazy module loading, and HF dataset response caching all amortize
during the opening rounds and would otherwise drag the slope into false
"climbing memory" flags.
"""
import json
import math
import os
import re
import sys
from pathlib import Path

STATS_RE = re.compile(r"^STATS (\{.*\})\s*$")
WARMUP_ROUNDS = int(os.environ.get("WARMUP_ROUNDS", "3"))


def linfit(xs, ys):
    """Least-squares slope/intercept. Returns (slope_per_unit_x, mean_y)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den > 0 else 0.0
    return slope, my


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def load(path_or_dash):
    out = []
    if path_or_dash == "-":
        src = sys.stdin
    else:
        src = open(path_or_dash)
    with src:
        for line in src:
            m = STATS_RE.match(line)
            if not m:
                continue
            try:
                out.append(json.loads(m.group(1)))
            except Exception:
                pass
    return out


def fmt_slope(slope, unit):
    sign = "+" if slope >= 0 else ""
    return f"{sign}{slope:.2f} {unit}/round"


def flag(cond, text):
    return f"  ⚠ {text}" if cond else None


def report(records, label):
    if not records:
        print(f"== {label} ==")
        print("  (no STATS records)")
        return

    rounds = [r["round"] for r in records]
    losses = [r["loss"] for r in records]
    elapsed = [r["elapsed_s"] for r in records]
    tok_s = [r["tok_s"] for r in records]
    gpu_mb = [r["gpu_mb"] for r in records]
    peak_mb = [r["peak_mb"] for r in records]
    pool_mb = [r["pool_mb"] for r in records]
    cpu_rss = [r["cpu_rss_mb"] for r in records]
    warns = [r["warns"] for r in records]
    warns_delta = [r["warns_delta"] for r in records]
    lags = [r["lag"] for r in records]
    contribs = [r["contributors"] for r in records]
    outer_steps = sum(1 for r in records if r.get("outer_step"))

    n = len(records)
    # Slopes computed on the post-warmup window so JS heap warmup, lazy
    # module loading, and HF response caching don't masquerade as leaks.
    warm_from = WARMUP_ROUNDS if n > WARMUP_ROUNDS + 1 else 0
    w_rounds = rounds[warm_from:]
    w_losses = losses[warm_from:]
    w_gpu = gpu_mb[warm_from:]
    w_rss = cpu_rss[warm_from:]
    w_pool = pool_mb[warm_from:]
    loss_slope, _ = linfit(w_rounds, w_losses)
    gpu_slope, gpu_mean = linfit(w_rounds, w_gpu)
    rss_slope, rss_mean = linfit(w_rounds, w_rss)
    pool_slope, pool_mean = linfit(w_rounds, w_pool)
    tok_sorted = sorted(tok_s)
    tok_p50 = quantile(tok_sorted, 0.5)
    tok_p95 = quantile(tok_sorted, 0.95)

    # Round-to-round loss noise (std of first-differences). Used to gate the
    # "loss trending up" flag so we don't false-positive on solo runs whose
    # loss naturally bobs without outer-step aggregation.
    loss_diffs = [w_losses[i + 1] - w_losses[i] for i in range(len(w_losses) - 1)]
    loss_noise = (sum(d * d for d in loss_diffs) / len(loss_diffs)) ** 0.5 if loss_diffs else 0.0

    flags = []
    # Slope thresholds: 5MB/round growth on a memory metric is a real leak
    # signal over a long run (50 rounds = 250MB). Loss is flagged only when
    # the upward slope is large compared to round-to-round noise — small
    # +slope on a noisy series is meaningless.
    flags.append(flag(
        loss_slope > 0.0 and loss_slope > 0.25 * loss_noise,
        f"loss is trending UP ({fmt_slope(loss_slope, 'loss')}, noise σ≈{loss_noise:.3f})"
    ))
    flags.append(flag(gpu_slope > 5.0, f"gpu_mb climbing ({fmt_slope(gpu_slope, 'MB')})"))
    flags.append(flag(rss_slope > 5.0, f"cpu_rss_mb climbing ({fmt_slope(rss_slope, 'MB')})"))
    flags.append(flag(pool_slope > 5.0, f"pool_mb climbing ({fmt_slope(pool_slope, 'MB')})"))
    flags.append(flag(warns[-1] > 0, f"Dawn warnings: {warns[-1]} total ({warns[-1] / n:.1f}/round avg)"))
    flags.append(flag(max(lags) > 5, f"peer-grad lag spiked to {max(lags)} (sustained lag >5 means a peer is struggling)"))
    flags = [f for f in flags if f]

    print(f"== {label} ==")
    warmup_note = f" (slopes ignore first {warm_from} rounds)" if warm_from else ""
    print(f"  rounds:       {n} (round {rounds[0]} .. {rounds[-1]}){warmup_note}")
    print(f"  time range:   {records[0]['t']} .. {records[-1]['t']}")
    print(f"  loss:         {losses[0]:.4f} -> {losses[-1]:.4f}  ({fmt_slope(loss_slope, 'loss')})")
    print(f"  elapsed_s:    p50={quantile(sorted(elapsed), 0.5):.1f}  p95={quantile(sorted(elapsed), 0.95):.1f}  total={sum(elapsed):.1f}s")
    print(f"  tok_s:        p50={tok_p50:.1f}  p95={tok_p95:.1f}")
    print(f"  gpu_mb:       mean={gpu_mean:.0f}  peak_max={max(peak_mb)}  ({fmt_slope(gpu_slope, 'MB')})")
    print(f"  pool_mb:      mean={pool_mean:.0f}  ({fmt_slope(pool_slope, 'MB')})")
    print(f"  cpu_rss_mb:   start={cpu_rss[0]}  end={cpu_rss[-1]}  ({fmt_slope(rss_slope, 'MB')})")
    print(f"  warns:        total={warns[-1]}  per-round mean={sum(warns_delta) / n:.2f}  max-in-round={max(warns_delta)}")
    print(f"  contributors: rounds_with_outer_step={outer_steps}/{n}  lag_max={max(lags)}")
    if flags:
        print("  flags:")
        for f in flags:
            print(f)
    else:
        print("  flags: (none)")


def main():
    sources = sys.argv[1:] or ["-"]
    for src in sources:
        label = "stdin" if src == "-" else Path(src).name
        records = load(src)
        report(records, label)


if __name__ == "__main__":
    main()
