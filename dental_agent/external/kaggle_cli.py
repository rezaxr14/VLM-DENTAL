"""The ``kaggle benchmarks`` command line, behind one small interface (the only place that parses its output).

Formats below were read from real output (probes v8/v9, CLI 2.2.5): ``tasks push`` prints ``Pushed [new version of] <task>`` and a
``.../<owner>/<task>/<version>`` URL, ``tasks run`` prints ``<model>: Scheduled``, ``tasks status`` prints a ``Version:`` line and a
``Model  Status  Started  Ended`` table, ``tasks download`` writes ``<out>/<task>/<version>/<model>/<run_id>/<files>``. A real
``push`` also starts one run on the account's default model; callers learn which from ``status``.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Runner = Callable[[list[str], float], tuple[int, str]]

STATUS_WORDS = ("completed", "running", "queued", "pending", "scheduled", "failed", "errored", "error", "cancelled", "canceled", "timeout")
ROW = re.compile(r"^(?P<model>[\w./:@-]+)\s+(?P<status>[A-Za-z_]+)\b(?P<rest>.*)$")
TERMINAL = {"completed", "failed", "errored", "error", "cancelled", "canceled", "timeout"}


def subprocess_runner(args: list[str], timeout: float) -> tuple[int, str]:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return -9, f"TIMEOUT after {timeout:.0f}s"
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if "outdated" not in ln.lower() and not ln.lower().startswith("warning")]
    return r.returncode, "\n".join(lines).strip()


class CLIError(RuntimeError):
    pass


@dataclass
class RunRow:
    model: str
    status: str                       # lower-case: completed | running | queued | failed | ...
    started: str = ""
    ended: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


@dataclass
class TaskStatus:
    task: str
    version: int | None
    rows: dict[str, RunRow] = field(default_factory=dict)
    raw: str = ""


def parse_status(task: str, text: str) -> TaskStatus:
    m = re.search(r"^Version:\s*(\d+)", text, re.M)
    rows: dict[str, RunRow] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or set(line) <= set("─-=| "):
            continue
        row = ROW.match(line)
        if not row or row["status"].lower() not in STATUS_WORDS or row["model"].endswith(":"):
            continue                                  # "Status:   Completed" is the task's own line, not a model row
        rest = row["rest"].split()
        started = " ".join(rest[:2]) if len(rest) >= 2 else ""
        ended = " ".join(rest[2:4]) if len(rest) >= 4 else ""
        rows[row["model"]] = RunRow(row["model"], row["status"].lower(), started, ended)
    return TaskStatus(task, int(m.group(1)) if m else None, rows, text)


class KaggleCLI:
    def __init__(self, runner: Runner | None = None):
        self._run = runner or subprocess_runner

    def _call(self, args: list[str], timeout: float = 300) -> tuple[int, str]:
        return self._run(["kaggle", "benchmarks", *args], timeout)

    def dataset_publish(self, folder: str | Path, dataset_id: str, message: str = "update") -> str:
        """Create the (private) dataset from ``folder``, or add a version when it already exists. Returns 'created' | 'versioned'."""
        rc, text = self._run(["kaggle", "datasets", "status", dataset_id], 120)
        exists = rc == 0 and "not found" not in text.lower() and "404" not in text
        args = ["kaggle", "datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "zip"] if exists else \
               ["kaggle", "datasets", "create", "-p", str(folder), "--dir-mode", "zip"]
        rc, text = self._run(args, 1800)
        if rc != 0 or re.search(r"\b(error|invalid|forbidden|unauthorized)\b", text, re.I):
            raise CLIError(f"dataset {dataset_id}: {text[:400]}")
        return "versioned" if exists else "created"

    def models(self) -> list[str]:
        rc, text = self._call(["tasks", "models"], 120)
        if rc != 0:
            raise CLIError(text[:300])
        out = []
        for line in text.splitlines():
            tok = line.split()[0] if line.split() else ""
            if tok and not set(tok) <= set("-=") and tok.lower() not in ("slug", "model", "display") and re.match(r"^[\w./:@-]+$", tok):
                out.append(tok)
        return list(dict.fromkeys(out))

    def quota(self) -> list[dict]:
        rc, text = self._call(["quota", "--format", "json"], 120)
        if rc != 0:
            raise CLIError(text[:300])
        return json.loads(text)

    def push(self, task: str, source: str | Path, datasets: list[str] = (), wait_s: int = 60) -> int | None:
        """Push (create or version) a task; returns its version. Raises ``CLIError`` with the CLI's own message on failure."""
        args = ["tasks", "push", task, "-f", str(source)]
        for d in datasets:
            args += ["-d", d]
        args += ["--wait", str(wait_s), "--poll-interval", "10"]
        rc, text = self._call(args, wait_s + 600)
        if rc != 0 or not re.search(r"\bPushed\b", text):
            raise CLIError(f"push {task}: {text[:500]}")
        m = re.search(rf"/{re.escape(task)}/(\d+)", text)
        return int(m.group(1)) if m else None

    def run(self, task: str, model: str) -> bool:
        """Schedule one run of ``task`` on ``model``. False (with no exception) when Kaggle refuses it."""
        rc, text = self._call(["tasks", "run", task, "-m", model], 300)
        return rc == 0 and "scheduled" in text.lower() and not re.search(r"\b(error|bad request|invalid|not found)\b", text, re.I)

    def run_text(self, task: str, model: str) -> tuple[bool, str]:
        rc, text = self._call(["tasks", "run", task, "-m", model], 300)
        ok = rc == 0 and "scheduled" in text.lower() and not re.search(r"\b(error|bad request|invalid|not found)\b", text, re.I)
        return ok, text[:400]

    def status(self, task: str) -> TaskStatus:
        rc, text = self._call(["tasks", "status", task], 180)
        if rc != 0:
            raise CLIError(f"status {task}: {text[:300]}")
        return parse_status(task, text)

    def download(self, task: str, out_dir: str | Path, model: str | None = None) -> Path:
        """Download finished run outputs; returns ``<out_dir>/<task>`` (layout ``<version>/<model>/<run_id>/<files>``)."""
        args = ["tasks", "download", task, "-o", str(out_dir)]
        if model:
            args += ["-m", model]
        args += ["-s", "-f"]
        rc, text = self._call(args, 900)
        if rc != 0:
            raise CLIError(f"download {task}: {text[:300]}")
        return Path(out_dir) / task
