"""
Hugging Face Hub artifact synchronization utilities.

Allows uploading and downloading checkpoints, logs, and trace datasets
to/from a private Hugging Face Model or Dataset repository across sessions.
"""

from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def sync_pull_artifacts(
    repo_id: str,
    local_dir: str | Path,
    repo_type: str = "model",
    token: str | None = None,
) -> Path:
    """Download artifacts from a Hugging Face repository to a local directory."""
    token = token or os.environ.get("HF_TOKEN")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        token=token,
    )
    return Path(path)


def sync_push_artifacts(
    local_dir: str | Path,
    repo_id: str,
    repo_type: str = "model",
    commit_message: str = "Sync experiment artifacts",
    token: str | None = None,
    private: bool = True,
) -> str:
    """Upload local directory artifacts to a Hugging Face repository."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be provided or set in the environment.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    return api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
