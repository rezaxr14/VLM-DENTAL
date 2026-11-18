"""Persistence of external-model results in the project's EXISTING private Hugging Face results repo.

Reuses ``harness.notebook_eval.HubResults`` (atomic bundles, identity-checked, optimistic concurrency, refuses to overwrite a
longer or conflicting copy) instead of adding another uploader. A bundle is one ``(model, condition, settings)`` cell, or one
shard of it when a hosted worker pushes its own slice, so several Kaggle accounts can contribute to the same cell without ever
touching the same bundle.

Only finished transport records are stored (answers, trajectories, per-request usage); ground truth never leaves this machine and
scoring is done locally on import, exactly as for a local run. No repository is ever created here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dental_agent.external.identity import fingerprint
from dental_agent.external.importer import completed, read_jsonl, safe_name

ROOT = "external"


def bundle_identity(settings: dict[str, Any], shard: str | None = None) -> dict[str, Any]:
    """The identity HubResults compares on restore/save: the run settings plus which slice this bundle holds."""
    return {"profile": settings["model_slug"], "kind": settings["condition"], "stage": "external", "shard": shard, "settings": settings}


def bundle_prefix(settings: dict[str, Any], shard: str | None = None) -> str:
    parts = [ROOT, settings.get("dataset", "dentex"), settings["split"], safe_name(settings["model_slug"]), settings["condition"],
             "run_" + fingerprint(settings)[:16]]
    if shard:
        parts.append(safe_name(shard))
    return "/".join(parts)


def result_rows(transport_path: str | Path) -> list[dict[str, Any]]:
    """Completed transport records, one per image (errors and superseded attempts are not persisted)."""
    return [r for _i, r in sorted(completed(read_jsonl(transport_path)).items())]


class ExternalHub:
    def __init__(self, hub):
        self.hub = hub                                     # a harness.notebook_eval.HubResults

    @classmethod
    def from_env(cls, repo: str | None = None, env=os.environ, repo_type: str | None = None) -> "ExternalHub | None":
        """None when there is no token or no repository configured (persistence is then simply off)."""
        repo = repo or env.get("HF_EVAL_REPO") or env.get("EVAL_RESULTS_REPO")
        token = env.get("HF_TOKEN")
        if not repo or not token or token.startswith("your_"):
            return None
        from huggingface_hub import HfApi, hf_hub_download
        from dental_agent.harness.notebook_eval import HubResults, result_repo_type
        api = HfApi(token=token)
        kind = result_repo_type(api, repo, repo_type or env.get("EVAL_RESULTS_REPO_TYPE"))
        return cls(HubResults(repo, api, hf_hub_download, token=token, repo_type=kind))

    def save_rows(self, folder: str | Path, prefix: str, identity: dict[str, Any], rows: list[dict[str, Any]], *, complete: bool) -> bool:
        """Push rows (each with an ``image_id``) as one atomic bundle under ``prefix``. False when there is nothing to push."""
        if not rows:
            return False
        payload = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows).encode()
        self.hub.save(folder, prefix, identity, {"results.jsonl": payload}, len(rows), complete)
        return True

    def save(self, folder: str | Path, settings: dict[str, Any], rows: list[dict[str, Any]], *, complete: bool, shard: str | None = None) -> bool:
        """Push transport ``rows`` of one cell (or shard) as one bundle. Returns False (and saves nothing) when there is nothing yet."""
        return self.save_rows(folder, bundle_prefix(settings, shard), bundle_identity(settings, shard), rows, complete=complete)

    def save_blob(self, folder: str | Path, prefix: str, identity: dict[str, Any], name: str, data: bytes, progress: int) -> None:
        """Push one non-result file (the campaign state) as an atomic bundle. ``progress`` must never decrease for the same prefix."""
        self.hub.save(folder, prefix, identity, {name: data}, int(progress), False)

    def load_blob(self, folder: str | Path, prefix: str, identity: dict[str, Any], name: str) -> bytes | None:
        folder = Path(folder)
        bundle = self.hub.restore(folder, prefix, identity)
        return None if bundle is None else (folder / name).read_bytes()

    def restore(self, folder: str | Path, settings: dict[str, Any], shard: str | None = None) -> list[dict[str, Any]]:
        """Rows already on the Hub for this cell/shard (identity-checked); ``[]`` when there is no bundle."""
        folder = Path(folder)
        bundle = self.hub.restore(folder, bundle_prefix(settings, shard), bundle_identity(settings, shard))
        if bundle is None:
            return []
        return [json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    def shards(self, settings: dict[str, Any]) -> list[str]:
        """Prefixes of every shard bundle of this cell that exists on the Hub."""
        base = bundle_prefix(settings) + "/"
        hub = self.hub
        info = hub.api.repo_info(hub.repo, repo_type=hub.repo_type)
        names = hub.api.list_repo_files(hub.repo, repo_type=hub.repo_type, revision=info.sha)
        return sorted({n.rsplit("/bundle.json", 1)[0] for n in names
                       if n.startswith(base) and n.endswith("/bundle.json") and n[len(base):] != "bundle.json"})   # shards only, not the cell's own bundle

    def restore_all_shards(self, folder: str | Path, settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
        """Merge every shard bundle (and the unsharded bundle) of a cell into ``{image_id: record}``; conflicting duplicates raise."""
        merged: dict[int, dict[str, Any]] = {}
        folder = Path(folder) / fingerprint(settings)[:16]          # one scratch folder per cell: two cells must never share restored files
        for prefix in [bundle_prefix(settings), *self.shards(settings)]:
            shard = None if prefix == bundle_prefix(settings) else prefix[len(bundle_prefix(settings)) + 1:]
            sub = folder / (shard or "_cell")
            rows = self.restore(sub, settings, shard)
            for r in rows:
                i = int(r["image_id"])
                if i in merged and merged[i] != r:
                    raise ValueError(f"image {i} has two different records on the Hub; neither is used")
                merged[i] = r
        return merged
