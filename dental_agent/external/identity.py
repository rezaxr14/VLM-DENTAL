"""Identity of an external-model run: everything that changes what its numbers mean. Stored in every record and checked on
resume, so a result can never be silently mixed with one produced under different settings."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from dental_agent.external.conditions import IMAGE_VIEW, LOOP_DEFAULTS, PROVIDER, Condition, max_turns


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cohort_sha(image_ids, cohort_settings: dict[str, Any] | None = None) -> str:
    """Hash of the image ids; on the 250-image cohort also of its COCO/mapping settings, so a new COCO version is a new cohort (new task names)."""
    ids = ",".join(str(i) for i in sorted(int(i) for i in image_ids))
    return sha256_text(ids + ("|" + json.dumps(cohort_settings, sort_keys=True) if cohort_settings else ""))


def prompt_hashes(condition: Condition, tool_descriptions: str | None = None, dataset: str = "dentex", split: str | None = None) -> dict[str, str]:
    """Hashes of the exact prompt text the model is given under ``condition`` (the cohort prompts on the 250-image split)."""
    from dental_agent.agent import prompts as P
    from dental_agent.external.cohort import prompts_for
    texts = prompts_for(split, dataset)
    out = {"first_user_prompt": sha256_text(P.FIRST_USER_PROMPT)}
    if condition.prompt == "legacy_zero_shot":
        out["zero_shot_prompt"] = sha256_text(texts.zero_shot)
    elif not condition.with_tools:
        out["system_prompt"] = sha256_text(texts.no_tools)
    else:
        if tool_descriptions is None:
            from dental_agent.external.tool_packs import get_registry
            tool_descriptions = get_registry(condition.pack).format_tool_descriptions()
        out["system_prompt"] = sha256_text(P.build_agent_system_prompt(tool_descriptions, dataset=texts.agent_dataset))
    return out


def run_settings(condition: Condition, model_slug: str, *, cohort: str, split: str, dataset: str = "dentex",
                 grounder_sha256: str | None = None, tool_descriptions: str | None = None,
                 overrides: dict[str, Any] | None = None, decoding: dict[str, Any] | None = None,
                 cohort_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    from dental_agent.external import cohort as C
    C.check_settings(split, cohort_settings)
    cfg = {**LOOP_DEFAULTS, **(overrides or {})}
    settings: dict[str, Any] = {
        "provider": PROVIDER, "model_slug": model_slug, "condition": condition.name, "track": condition.track,
        "dataset": dataset, "split": split, "cohort_sha256": cohort,
        "canonical_resize": IMAGE_VIEW["canonical_resize"], "canonical_letterbox": IMAGE_VIEW["letterbox"],
        "max_tool_calls": int(cfg["max_tool_calls"]) if condition.with_tools else 0,
        "max_turns": max_turns(cfg["max_tool_calls"]) if condition.with_tools else 1,
        "max_new_tokens": int(cfg["max_new_tokens"]), "malformed_retries": int(cfg["malformed_retries"]),
        "temperature": float(cfg["temperature"]), "retry_policy": "fail_fast_no_retry",
        "prompt": condition.prompt, "prompt_sha256": prompt_hashes(condition, tool_descriptions, dataset, split),
    }
    if cohort_settings:                           # 250-image cohort: COCO/mapping hashes, scoring profile, prompt classes (as eval_settings)
        settings[C.SETTINGS_KEY] = dict(cohort_settings)
    if decoding is not None:                      # per-model facts (e.g. temperature not sent), from model_profiles
        settings["decoding"] = decoding
    if condition.pack != "default":               # a non-default tool pack is part of what the numbers mean
        settings["pack"] = condition.pack
    if condition.evaluator != "dentex_findings":
        settings["evaluator"] = condition.evaluator
    if condition.with_tools:
        settings["grounder_sha256"] = grounder_sha256
        from dental_agent.harness.configuration import harness_identity
        settings["harness"] = harness_identity(condition.harness_mode)
    return settings


def fingerprint(settings: dict[str, Any]) -> str:
    return sha256_text(json.dumps(settings, sort_keys=True, default=str))


class IdentityMismatch(ValueError):
    pass


def assert_same(saved: dict[str, Any] | None, current: dict[str, Any]) -> None:
    """Resume guard. Lists the differing keys so the operator sees what changed."""
    saved = saved or {}
    keys = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
    if keys:
        raise IdentityMismatch("settings differ from the saved run in: " + ", ".join(keys))
