"""Checkpoint metadata and Hub destinations; no model loading or precision changes."""
from __future__ import annotations

import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any


def validate_base_model_id(value: str) -> str:
    from huggingface_hub.utils import validate_repo_id
    if not isinstance(value, str):
        raise ValueError("Hub base model ID must be a string")
    value = value.strip()
    validate_repo_id(value)
    if value.startswith(("/", "\\", "~")) or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Hub base model must be a model ID, not a local filesystem path")
    return value


def sft_hub_subfolder(track: str, stage: str, precision: str, run_tag: str | None = None) -> str:
    from dental_agent.utils.inputs import sft_adapter_name
    for value in (track, stage, precision):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]+", value):
            raise ValueError("Track, stage and precision must be safe artifact-name components")
    name = sft_adapter_name(track, stage, precision)
    if run_tag:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_tag) or ".." in run_tag:
            raise ValueError("--hf-run-tag must be a single name using letters, numbers, _, - or .")
        name += "__" + run_tag
    return "sft/" + name


def _valid_card_base(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(_valid_card_base(v) for v in value)
    if not isinstance(value, str) or not value:
        return False
    try:
        validate_base_model_id(value)
    except ValueError:
        return False
    # Relative local paths can look syntactically like a Hub ID.
    return not Path(value).exists()


def _inside(path: Path, root: Path) -> Path:
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Checkpoint metadata points outside the checkpoint folder: {path}")
    return path


def _temporary(path: Path) -> Path:
    # Unique files avoid a collision with a concurrent preemption checkpoint.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False) as f:
        return Path(f.name)


def prepare_hub_metadata(folder: str | Path, base_model_id: str | None = None) -> dict[str, Any]:
    """Repair root and nested PEFT cards, including best_adapter and GRPO adapters.

    An explicit Hub ID also makes adapter_config.json portable. Without one,
    preserve valid existing metadata and omit invalid optional base_model fields
    rather than inventing an origin. Never rewrite the in-memory model config.
    """
    from huggingface_hub import ModelCard, ModelCardData
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint folder not found: {root}")
    base = validate_base_model_id(base_model_id) if base_model_id else None
    configs = sorted(root.rglob("adapter_config.json"))
    saved_bases = {}
    cards = {p.parent / "README.md" for p in configs}
    if (root / "README.md").exists():
        cards.add(root / "README.md")
    repaired = 0
    for path in configs:
        _inside(path, root)
        data = json.loads(path.read_text(encoding="utf-8"))
        candidate = data.get("base_model_name_or_path")
        if isinstance(candidate, str) and _valid_card_base(candidate):
            saved_bases[path.parent] = candidate
        if base:
            if data.get("base_model_name_or_path") != base:
                data["base_model_name_or_path"] = base
                temp = _temporary(path)
                temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                temp.replace(path)
    for path in sorted(cards):
        _inside(path, root)
        card = ModelCard.load(str(path)) if path.exists() else ModelCard.from_template(ModelCardData(library_name="peft"))
        metadata = card.data.to_dict()
        inferred = saved_bases.get(path.parent)
        if inferred is None and len(set(saved_bases.values())) == 1:
            inferred = next(iter(saved_bases.values()))
        if base:
            metadata["base_model"] = base
        elif "base_model" in metadata and not _valid_card_base(metadata["base_model"]):
            metadata.pop("base_model")
        if "base_model" not in metadata and inferred:
            metadata["base_model"] = inferred
        card.data = ModelCardData(**metadata)
        temp = _temporary(path)
        card.save(str(temp))
        temp.replace(path)
        repaired += 1
    return {"base_model_id": base, "model_cards": repaired, "adapter_configs": len(configs)}


def upload_adapter_folder(folder: str | Path, repo_id: str, path_in_repo: str | None = None,
                          *, base_model_id: str | None = None, commit_message: str = "Upload adapter checkpoint"):
    from huggingface_hub import HfApi
    if not any(Path(folder).rglob("adapter_config.json")):
        raise ValueError(f"No adapter_config.json in checkpoint folder: {folder}")
    prepare_hub_metadata(folder, base_model_id)
    kwargs = dict(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                  commit_message=commit_message, ignore_patterns=["*.tmp", "*.lock"])
    if path_in_repo:
        kwargs["path_in_repo"] = path_in_repo
    return HfApi().upload_folder(**kwargs)


def check_hub_write_access(repo_id: str, path_in_repo: str, *, metadata: dict[str, Any]):
    """Small authenticated write before weight loading; never called by --preflight."""
    from huggingface_hub import HfApi
    payload = {"purpose": "pre-training upload permission check; not a model checkpoint", **metadata}
    try:
        return HfApi().upload_file(
            path_or_fileobj=io.BytesIO((json.dumps(payload, indent=2) + "\n").encode()),
            path_in_repo=path_in_repo + "/upload_check.json", repo_id=repo_id, repo_type="model",
            commit_message="Verify checkpoint upload access before training",
        )
    except Exception as exc:
        raise RuntimeError(
            f"[HF-HUB PRECHECK FAILED] Cannot write to {repo_id}/{path_in_repo}: {exc}. "
            "No model weights have been loaded. Fix token/repository access, or explicitly use --no-hf-upload-check."
        ) from exc
