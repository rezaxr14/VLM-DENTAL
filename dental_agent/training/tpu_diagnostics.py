"""Opt-in diagnostics for unattended TPU runs (Kaggle commit runs have no interactive session).

Off unless ``--tpu-report-dir`` is given: importing this module has no side effects, nothing is created, and
``torch_xla`` is never imported by this module. It only *reads* XLA state once the launcher has already imported
``torch_xla.core.xla_model`` (importing it initializes libtpu, which the launcher alone decides when to do).

What a finished or dead run leaves behind (all under ``--tpu-report-dir``, flushed as it happens):

* ``environment.json``  package versions (read from metadata, no imports), selected env vars, host RAM, argv.
* ``events.jsonl``      begin/end/mark events with seconds, host RSS, XLA compile-count deltas and device memory.
* ``snapshots.jsonl``   XLA compile count/time, ``aten::`` CPU-fallback counters and device memory at selected steps.
* ``heartbeat.jsonl``   host RSS / available RAM / current stage and step every N seconds (read-only, no XLA calls).
* ``tpu_info.log``      periodic output of the ``tpu-info`` CLI (true per-chip HBM), when that CLI exists.
* ``stacks.txt``        Python stacks of every thread: on a native abort, on SIGTERM, and every N seconds while the
                        process is alive (so a multi-hour compile or hang shows where the main thread is blocked).
* ``alerts.txt``        plain-text warnings, e.g. XLA recompiled after warm-up.
* ``xla_metrics_*.txt`` full ``torch_xla`` metrics report at failure/finish.
* ``final.json``        outcome: ok / crashed (traceback, stage) / terminated.

The report folder can be pushed to an existing Hub repo (never created here) while the run is still alive, so a
session that is killed hours in still leaves evidence. All reporting is best-effort and never raises into training.
"""
from __future__ import annotations

import atexit
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = ("torch", "torch_xla", "libtpu", "libtpu-nightly", "jax", "jaxlib", "tpu-info", "transformers", "peft",
            "accelerate", "huggingface_hub", "numpy", "Pillow", "bitsandbytes")
ENV_VARS = ("PJRT_DEVICE", "XLA_FLAGS", "LIBTPU_INIT_ARGS", "XLA_PERSISTENT_CACHE_PATH", "XLA_USE_SPMD",
            "XLA_HLO_DEBUG", "XLA_IR_DEBUG", "TPU_NAME", "TPU_ACCELERATOR_TYPE", "TPU_CHIPS_PER_HOST_BOUNDS",
            "TPU_HOST_BOUNDS", "CLOUD_TPU_TASK_ID", "TPU_PROCESS_COUNT", "KAGGLE_KERNEL_RUN_TYPE", "HF_HOME",
            "TOKENIZERS_PARALLELISM", "PYTHONUNBUFFERED", "MALLOC_ARENA_MAX")
UPLOAD_PATTERNS = ("*.json", "*.jsonl", "*.txt", "*.log")

_ACTIVE: "TPUReport | None" = None


def add_report_arguments(parser) -> None:
    """Explicit, notebook-visible controls. ``--tpu-report-dir`` unset means the whole feature is off."""
    group = parser.add_argument_group("TPU run report (diagnostics for unattended runs; off unless a directory is given)")
    group.add_argument("--tpu-report-dir", default=None, help="Directory for the run report; unset disables every feature below.")
    group.add_argument("--tpu-report-heartbeat-seconds", type=float, default=60.0, help="Heartbeat period (host-only).")
    group.add_argument("--tpu-report-stack-seconds", type=float, default=900.0,
                       help="Dump every thread's Python stack this often while alive; 0 disables the periodic dump.")
    group.add_argument("--tpu-report-tpu-info-seconds", type=float, default=300.0,
                       help="Run the tpu-info CLI (per-chip HBM) this often; 0 disables.")
    group.add_argument("--tpu-report-step-snapshot-every", type=int, default=25,
                       help="XLA snapshot every N optimizer steps (steps 1-3 are always snapshotted).")
    group.add_argument("--tpu-report-hub-repo", default=None, help="Existing Hub repo to push the report to (never created here).")
    group.add_argument("--tpu-report-hub-repo-type", default="model", choices=["model", "dataset"])
    group.add_argument("--tpu-report-hub-subfolder", default=None, help="Folder inside that repo, e.g. tpu_reports/sft/run1.")
    group.add_argument("--tpu-report-upload-seconds", type=float, default=600.0, help="Push period while alive; 0 pushes only at the end.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _kib(line: str) -> float:
    return int(line.split()[1]) / (1024 ** 2)


def host_memory() -> dict[str, float]:
    """Host RSS / peak RSS / available RAM in GiB from /proc (Linux); empty elsewhere."""
    out: dict[str, float] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                out["rss_gib"] = round(_kib(line), 3)
            elif line.startswith("VmHWM:"):
                out["peak_rss_gib"] = round(_kib(line), 3)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                out["host_available_gib"] = round(_kib(line), 3)
            elif line.startswith("MemTotal:"):
                out["host_total_gib"] = round(_kib(line), 3)
    except OSError:
        pass
    return out


def _xla_ready() -> bool:
    # The launcher imports xla_model exactly when it is ready to claim the device; never do that for it.
    return "torch_xla.core.xla_model" in sys.modules


def compile_stats() -> dict[str, Any]:
    """XLA compilations so far: count and the accumulated time as reported by torch_xla (nanoseconds)."""
    try:
        import torch_xla.debug.metrics as met
        data = met.metric_data("CompileTime")
        return {"count": int(data[0]), "total": float(data[1])} if data else {"count": 0, "total": 0.0}
    except Exception as exc:  # noqa: BLE001 — diagnostics must not break training
        return {"error": f"{type(exc).__name__}: {exc}"}


def fallback_counters() -> dict[str, int]:
    """Ops that fell back to the CPU (``aten::*`` counters) — each is a host round trip and a hidden slowdown."""
    try:
        import torch_xla.debug.metrics as met
        return {n: int(met.counter_value(n)) for n in met.counter_names() if n.startswith("aten::")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}  # type: ignore[dict-item]


def device_memory() -> dict[str, Any]:
    """``xm.get_memory_info`` for this process's device (chip 0 under SPMD); tpu-info covers every chip."""
    try:
        import torch_xla.core.xla_model as xm
        info = xm.get_memory_info(xm.xla_device())
        return {k: (round(v / 1024 ** 3, 3) if isinstance(v, (int, float)) else v) for k, v in dict(info).items()} | {"unit": "GiB"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _upload_folder(folder: Path, repo_id: str, repo_type: str, path_in_repo: str | None, message: str) -> None:
    from huggingface_hub import HfApi
    HfApi().upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=repo_type, path_in_repo=path_in_repo,
                          allow_patterns=list(UPLOAD_PATTERNS), commit_message=message)


class TPUReport:
    def __init__(self, directory: str | Path, *, role: str, argv: list[str] | None = None,
                 heartbeat_seconds: float = 60.0, stack_seconds: float = 900.0, tpu_info_seconds: float = 300.0,
                 step_snapshot_every: int = 25, hub_repo: str | None = None, hub_repo_type: str = "model",
                 hub_subfolder: str | None = None, upload_seconds: float = 600.0):
        for name, value in (("heartbeat", heartbeat_seconds), ("stack", stack_seconds), ("tpu-info", tpu_info_seconds),
                            ("upload", upload_seconds)):
            if value < 0:
                raise ValueError(f"--tpu-report-{name}-seconds must be nonnegative")
        if step_snapshot_every < 1:
            raise ValueError("--tpu-report-step-snapshot-every must be at least 1")
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.role, self.argv = role, list(argv if argv is not None else sys.argv)
        self.heartbeat_seconds, self.stack_seconds = heartbeat_seconds, stack_seconds
        self.tpu_info_seconds, self.step_snapshot_every = tpu_info_seconds, step_snapshot_every
        self.hub_repo, self.hub_repo_type, self.hub_subfolder = hub_repo, hub_repo_type, hub_subfolder
        self.upload_seconds = upload_seconds
        self.stage: str | None = None
        self.step_number = 0
        self.fields: dict[str, Any] = {}
        self._begun: dict[str, tuple[float, dict[str, Any]]] = {}
        self._last_mark = time.monotonic()
        self._baseline_compiles: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stack_file = None
        self._finished = False
        self._previous_hook = None

    # ----- construction -------------------------------------------------------------------------------------
    @classmethod
    def from_args(cls, args, role: str) -> "TPUReport | None":
        directory = getattr(args, "tpu_report_dir", None)
        if not directory:
            return None
        return cls(directory, role=role, heartbeat_seconds=args.tpu_report_heartbeat_seconds,
                   stack_seconds=args.tpu_report_stack_seconds, tpu_info_seconds=args.tpu_report_tpu_info_seconds,
                   step_snapshot_every=args.tpu_report_step_snapshot_every, hub_repo=args.tpu_report_hub_repo,
                   hub_repo_type=args.tpu_report_hub_repo_type, hub_subfolder=args.tpu_report_hub_subfolder,
                   upload_seconds=args.tpu_report_upload_seconds)

    # ----- files ----------------------------------------------------------------------------------------------
    def _write_json(self, name: str, value: Any) -> None:
        try:
            temp = self.dir / f".{name}.tmp"
            temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            temp.replace(self.dir / name)
        except Exception as exc:  # noqa: BLE001
            print(f"[TPU REPORT] could not write {name}: {exc}", flush=True)

    def _append(self, name: str, value: Any) -> None:
        try:
            with self._lock, open(self.dir / name, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:  # noqa: BLE001
            print(f"[TPU REPORT] could not append to {name}: {exc}", flush=True)

    def _text(self, name: str, text: str, mode: str = "a") -> None:
        try:
            with self._lock, open(self.dir / name, mode, encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
        except Exception as exc:  # noqa: BLE001
            print(f"[TPU REPORT] could not write {name}: {exc}", flush=True)

    def alert(self, message: str) -> None:
        print(f"[TPU REPORT ALERT] {message}", flush=True)
        self._text("alerts.txt", f"{_now()} {message}\n")

    # ----- start ---------------------------------------------------------------------------------------------
    def start(self) -> "TPUReport":
        global _ACTIVE
        _ACTIVE = self
        self.environment()
        self.install_crash_capture()
        if self.heartbeat_seconds > 0:
            self._thread = threading.Thread(target=self._beat, name="tpu-report-heartbeat", daemon=True)
            self._thread.start()
        atexit.register(self.finish)
        self.mark("report_started")
        return self

    def environment(self) -> dict[str, Any]:
        versions = {}
        for name in PACKAGES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        try:
            free = round(shutil.disk_usage(self.dir).free / 1024 ** 3, 1)
        except OSError:
            free = None
        info = {"role": self.role, "started": _now(), "argv": self.argv, "python": sys.version, "cwd": os.getcwd(),
                "versions": versions, "env": {k: os.environ.get(k) for k in ENV_VARS},
                "host_memory_gib": host_memory(), "cpu_count": os.cpu_count(), "report_dir_free_gib": free,
                "tpu_info_cli": shutil.which("tpu-info")}
        self._write_json("environment.json", info)
        return info

    def install_crash_capture(self) -> None:
        try:
            self._stack_file = open(self.dir / "stacks.txt", "a", encoding="utf-8")
            faulthandler.enable(file=self._stack_file, all_threads=True)  # native aborts (XLA CHECK failures, SIGSEGV)
            if self.stack_seconds > 0:
                faulthandler.dump_traceback_later(self.stack_seconds, repeat=True, file=self._stack_file)
            if hasattr(faulthandler, "register"):
                # C-level: still fires while the main thread sits inside a long native XLA compile or execute.
                faulthandler.register(signal.SIGTERM, file=self._stack_file, all_threads=True, chain=True)
                faulthandler.register(signal.SIGUSR1, file=self._stack_file, all_threads=True, chain=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[TPU REPORT] crash capture partially unavailable: {exc}", flush=True)
        self._previous_hook = sys.excepthook

        def hook(kind, value, tb):
            self.crash((kind, value, tb))
            (self._previous_hook or sys.__excepthook__)(kind, value, tb)

        sys.excepthook = hook

    # ----- events --------------------------------------------------------------------------------------------
    def _context(self) -> dict[str, Any]:
        context: dict[str, Any] = {"t": _now(), "stage": self.stage, "step": self.step_number, "host": host_memory()}
        if _xla_ready():
            context["compile"] = compile_stats()
            context["device_memory"] = device_memory()
        return context

    def mark(self, name: str, **meta: Any) -> None:
        now = time.monotonic()
        record = {"event": "mark", "name": name, "since_previous_mark_s": round(now - self._last_mark, 2), **meta}
        self._last_mark = now
        self._append("events.jsonl", {**record, **self._context()})

    def begin(self, name: str, **meta: Any) -> None:
        self.stage = name
        self._begun[name] = (time.monotonic(), compile_stats() if _xla_ready() else {})
        self._append("events.jsonl", {"event": "begin", "name": name, **meta, **self._context()})

    def end(self, name: str, **meta: Any) -> None:
        started, before = self._begun.pop(name, (time.monotonic(), {}))
        record: dict[str, Any] = {"event": "end", "name": name, "seconds": round(time.monotonic() - started, 2), **meta}
        if _xla_ready():
            after = compile_stats()
            if "count" not in before:
                before = {"count": 0, "total": 0.0}  # XLA did not exist yet when the stage began: nothing was compiled
            if "count" in before and "count" in after:
                record["compiles_in_stage"] = after["count"] - before["count"]
                record["compile_time_in_stage_s"] = round((after["total"] - before["total"]) / 1e9, 2)
        self._append("events.jsonl", {**record, **self._context()})
        if self.stage == name:
            self.stage = None

    def step(self, number: int, **fields: Any) -> None:
        self.step_number, self.fields = number, fields
        if not _xla_ready() or not (number <= 3 or number % self.step_snapshot_every == 0):
            return
        stats = compile_stats()
        snapshot = {"step": number, "fields": fields, "compile": stats, "fallbacks": fallback_counters(),
                    "device_memory": device_memory(), "host": host_memory(), "t": _now()}
        count = stats.get("count")
        if number == 3 and isinstance(count, int):
            self._baseline_compiles = count
        elif number > 3 and self._baseline_compiles is not None and isinstance(count, int) and count > self._baseline_compiles:
            snapshot["recompiled_after_warmup"] = count - self._baseline_compiles
            self.alert(f"XLA compiled {count - self._baseline_compiles} new graph(s) after step 3 (now at step {number}); "
                       "a shape or Python value is changing between steps.")
            self._baseline_compiles = count
        self._append("snapshots.jsonl", snapshot)

    # ----- background ------------------------------------------------------------------------------------------
    def tpu_info(self) -> str | None:
        executable = shutil.which("tpu-info")
        if not executable:
            return None
        try:
            done = subprocess.run([executable], capture_output=True, text=True, timeout=30)
            return done.stdout or done.stderr
        except Exception as exc:  # noqa: BLE001
            return f"tpu-info failed: {type(exc).__name__}: {exc}"

    def _beat(self) -> None:
        last_info = last_upload = time.monotonic()
        while not self._stop.wait(self.heartbeat_seconds):
            self._append("heartbeat.jsonl", {"t": _now(), "stage": self.stage, "step": self.step_number,
                                             "fields": self.fields, "host": host_memory()})
            now = time.monotonic()
            if self.tpu_info_seconds and now - last_info >= self.tpu_info_seconds:
                last_info = now
                text = self.tpu_info()
                if text:
                    self._text("tpu_info.log", f"===== {_now()} stage={self.stage} step={self.step_number}\n{text}\n")
            if self.hub_repo and self.upload_seconds and now - last_upload >= self.upload_seconds:
                last_upload = now
                self.upload(f"{self.role} report: heartbeat, stage={self.stage}, step={self.step_number}")

    def upload(self, message: str) -> bool:
        if not self.hub_repo:
            return False
        try:
            _upload_folder(self.dir, self.hub_repo, self.hub_repo_type, self.hub_subfolder, message)
            return True
        except Exception as exc:  # noqa: BLE001 — a flaky network must not stop or crash training
            self._text("upload_errors.txt", f"{_now()} {type(exc).__name__}: {exc}\n")
            return False

    # ----- end -------------------------------------------------------------------------------------------------
    def _metrics_text(self, tag: str) -> None:
        if not _xla_ready():
            return
        try:
            import torch_xla.debug.metrics as met
            self._text(f"xla_metrics_{tag}.txt", met.metrics_report(), mode="w")
        except Exception as exc:  # noqa: BLE001
            self._text(f"xla_metrics_{tag}.txt", f"unavailable: {type(exc).__name__}: {exc}\n", mode="w")

    def _finalize(self, status: str, extra: dict[str, Any]) -> None:
        if self._finished:
            return
        self._finished = True
        self._stop.set()
        record = {"status": status, "role": self.role, "stage": self.stage, "step": self.step_number,
                  "fields": self.fields, **extra, **self._context()}
        if _xla_ready():
            record["fallbacks"] = fallback_counters()
        self._write_json("final.json", record)
        self._metrics_text("final")
        info = self.tpu_info() if self.tpu_info_seconds else None
        if info:
            self._text("tpu_info.log", f"===== {_now()} final status={status}\n{info}\n")
        self.upload(f"{self.role} report: {status}")

    def crash(self, exc_info) -> None:
        kind, value, tb = exc_info
        self._finalize("crashed", {"error": f"{getattr(kind, '__name__', kind)}: {value}",
                                   "traceback": "".join(traceback.format_exception(kind, value, tb))})

    def finish(self, status: str = "ok") -> None:
        if status == "ok":
            self.stage = None  # a clean exit has no 'current' stage; crashes keep it
        self._finalize(status, {})


# ----- module-level helpers: no-ops unless a report was started -------------------------------------------------
def start_report(args, role: str) -> TPUReport | None:
    report = TPUReport.from_args(args, role)
    return report.start() if report else None


def mark(name: str, **meta: Any) -> None:
    if _ACTIVE:
        _ACTIVE.mark(name, **meta)


def begin(name: str, **meta: Any) -> None:
    if _ACTIVE:
        _ACTIVE.begin(name, **meta)


def end(name: str, **meta: Any) -> None:
    if _ACTIVE:
        _ACTIVE.end(name, **meta)


def step(number: int, **fields: Any) -> None:
    if _ACTIVE:
        _ACTIVE.step(number, **fields)


def finish(status: str = "ok") -> None:
    if _ACTIVE:
        _ACTIVE.finish(status)
