#!/usr/bin/env python3
"""Summarise a GRPO step log and project the run time of a full pool and of other group sizes.

Probe first, then decide how long the real run takes. Example (10 images, K=4, written to its own log):

    python scripts/run_grpo.py --track with_tools --group-size 4 --max-images 10 --log-path data/grpo_probe_k4.jsonl ...
    python scripts/summarize_grpo_log.py --log data/grpo_probe_k4.jsonl --pool-size 635 --project-k 2 4 8 16

Projection: seconds per step = rollout_s + update_s, averaged after the first ``--skip-first`` steps (XLA compiles
there, and the first rollouts warm the KV cache). Both parts grow linearly with K (K rollouts, K trajectories in the
update), so another K scales the measured step time by K / K_measured. Worked example: a probe at K=4 averaging 600 s per
step over 635 pool images is 635 x 600 s = 106 h; K=8 projects to 211 h. It is an estimate from the probe's images, and
the mean turn / tool-call counts printed beside it show whether those images were typical.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_log(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict], pool_size: int, skip_first: int = 2, project_k: list[int] | None = None) -> dict:
    timed = [r for r in rows if "rollout_s" in r and "update_s" in r]
    if not timed:
        raise SystemExit("The log has no rollout_s/update_s fields: it was written before per-step timing existed. "
                         "Run a probe with the current run_grpo.py.")
    timed.sort(key=lambda r: r.get("step", 0))
    used = timed[skip_first:] if len(timed) > skip_first else timed
    ks = {int(r["group_size"]) for r in used if "group_size" in r}
    if len(ks) != 1:
        raise SystemExit(f"The log mixes group sizes {sorted(ks)}; summarise one run (one --log-path) at a time.")
    k = ks.pop()

    step_s = [r["rollout_s"] + r["update_s"] for r in used]
    mean_step = statistics.fmean(step_s)
    out = {
        "steps_logged": len(timed),
        "steps_used": len(used),
        "group_size": k,
        "mean_rollout_s": statistics.fmean(r["rollout_s"] for r in used),
        "mean_update_s": statistics.fmean(r["update_s"] for r in used),
        "mean_step_s": mean_step,
        "p90_step_s": sorted(step_s)[min(len(step_s) - 1, int(0.9 * len(step_s)))],
        "mean_turns": statistics.fmean(r.get("mean_turns", 0.0) for r in used),
        "mean_tool_calls": statistics.fmean(r.get("mean_tool_calls", 0.0) for r in used),
        "format_ok_rate": statistics.fmean(r.get("format_ok_rate", 0.0) for r in used),
        "mean_reward": statistics.fmean(r["mean_reward"] for r in used if "mean_reward" in r),
        "dropped_overlength": sum(int(r.get("n_dropped_overlength", 0)) for r in used),
        "pool_size": pool_size,
        "projection_hours": {},
    }
    for kk in sorted({k, *(project_k or [])}):
        out["projection_hours"][kk] = pool_size * mean_step * (kk / k) / 3600.0
    return out


def format_summary(s: dict) -> str:
    lines = [
        f"Steps logged {s['steps_logged']} (used {s['steps_used']} after the warm-up), group size K={s['group_size']}",
        f"Per step: rollout {s['mean_rollout_s']:.0f} s + update {s['mean_update_s']:.0f} s = {s['mean_step_s']:.0f} s "
        f"(p90 {s['p90_step_s']:.0f} s)",
        f"Rollouts: {s['mean_turns']:.1f} turns, {s['mean_tool_calls']:.1f} tool calls per trajectory, "
        f"format ok {s['format_ok_rate']:.0%}, {s['dropped_overlength']} dropped over-length",
        f"Mean reward {s['mean_reward']:.3f}",
        f"Projection for {s['pool_size']} pool images (one image per step):",
    ]
    for kk, hours in s["projection_hours"].items():
        tag = "measured K" if kk == s["group_size"] else "scaled linearly in K"
        lines.append(f"  K={kk:<3d} {hours:8.1f} h   ({tag})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise a GRPO step log and project total run time.")
    ap.add_argument("--log", required=True, help="JSONL written by run_grpo.py --log-path")
    ap.add_argument("--pool-size", type=int, required=True, help="Number of training images of the real run (= steps)")
    ap.add_argument("--skip-first", type=int, default=2, help="Warm-up steps excluded from the averages (XLA compile)")
    ap.add_argument("--project-k", type=int, nargs="*", default=[], help="Group sizes to project to")
    args = ap.parse_args()
    print(format_summary(summarize(load_log(args.log), args.pool_size, args.skip_first, args.project_k)))


if __name__ == "__main__":
    main()
