"""The exact trained checkpoint an external benchmark is run with: resolved by name, never substituted."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def grpo_policy_location(repo: str, track: str, k: int, stage: str, precision: str, tag: str, harness_mode: str) -> dict[str, str]:
    """Hub prefix and local folder of one GRPO run. Names come from ``experiment_profiles.grpo_name`` (the training side's own function)."""
    from dental_agent.training.experiment_profiles import grpo_name
    name = grpo_name(track, k, stage, precision, tag, harness_mode)
    return dict(name=name, hub_prefix=f"grpo/{name}/grpo-{track}-final", local_root=f"data/models/{name}/grpo-{track}-final")


def fetch_grpo_policy(repo: str, track: str, k: int, stage: str, precision: str, tag: str, harness_mode: str,
                      local_models: str | Path = "data/models") -> dict[str, Any]:
    """Folder of ``grpo_policy`` for exactly this run, from this server's ``data/models`` if training happened here, else from the
    models Hub repository. A missing or incomplete checkpoint raises; no other K, harness mode, stage or tag is ever used instead."""
    loc = grpo_policy_location(repo, track, k, stage, precision, tag, harness_mode)
    final = Path(local_models) / loc["name"] / f"grpo-{track}-final"
    source = "local"
    if not (final / "grpo_policy" / "adapter_config.json").is_file():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo_id=repo, repo_type="model", allow_patterns=[f"{loc['hub_prefix']}/*"]))
        final = root / loc["hub_prefix"]
        source = f"hub:{repo}"
    policy = final / "grpo_policy"
    if not (policy / "adapter_config.json").is_file() or not any(policy.glob("adapter_model*")):
        raise FileNotFoundError(f"Missing exact GRPO policy: {repo}/{loc['hub_prefix']}/grpo_policy (also not under {local_models}). "
                                "Nothing else is substituted: train it, or fix K / harness mode / stage / profile tag.")
    meta_file = final / "checkpoint_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else None
    return dict(path=str(policy), name=loc["name"], source=source, meta=meta, has_meta=meta is not None)
