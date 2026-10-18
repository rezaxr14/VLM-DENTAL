#!/usr/bin/env python3
"""Decision logic and verification helpers for ``scripts/setup_gpu_server.sh`` (patch 0039).

Why this is a module and not shell: the choices (which torch wheel index, which attention backend, what a ``.env`` must look
like) are pure functions of facts read from the machine, so they are unit-tested without a GPU (``tests/test_gpu_setup.py``).
The shell stays thin: it runs commands and calls this file.

Import rules
------------
* Standard library only at import time. The bootstrap runs this file BEFORE anything is installed, by path
  (``python3 dental_agent/utils/gpu_setup.py report``), because ``import dental_agent`` pulls in torch, pandas and friends.
  It must also run on the system Python 3.10 of Ubuntu 22.04.
* The ``verify-*``, ``env-tags``, ``fetch-data`` and ``write-constraints`` commands import torch / the project lazily; the shell
  runs them with the venv's Python after the install steps.

What it never does
------------------
It never changes a training setting (precision, sequence length, LoRA, hyperparameters, notebook cells), never commits or pushes,
never deletes or kills anything, never prints a token value. Precision only gets a printed hint.

Facts the tables below rest on (looked up on 2026-10-18, not assumed):
* PyTorch 2.12 / 2.13 publish CUDA builds cu126, cu130 and cu132 only (pytorch.org "previous versions"); 2.11.0 is the last release
  listed with cu128. cu132 is experimental.
* NVIDIA's CUDA 13.0 notes: Linux driver >= 580.65.06 (GA). CUDA 12.8 GA: >= 570.117. CUDA 12.x minor-version compatibility: >= 525.
* nvcc 12.8 is the first that targets sm_100 / sm_101 / sm_120, so a Blackwell GPU needs a torch built with CUDA >= 12.8.
* FlashAttention-2 supports Ampere, Ada and Hopper; FlashAttention-3 is a Hopper (H100/H800) beta; FlashAttention-4 targets
  Hopper and B200-class Blackwell. For sm_120 the upstream tracker (Dao-AILab/flash-attention issue 1987) shows no prebuilt wheel
  and no maintainer statement of support.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


class SetupError(RuntimeError):
    """A decision that cannot be made safely. The message says what to do about it."""


# --------------------------------------------------------------------------------------------------------------------
# Versions and nvidia-smi parsing
# --------------------------------------------------------------------------------------------------------------------
def version_tuple(text: str) -> tuple[int, ...]:
    """``'570.172.08'`` -> ``(570, 172, 8)``; compares correctly as a tuple."""
    return tuple(int(x) for x in re.findall(r"\d+", text or ""))


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_mib: int
    cc: tuple[int, int]
    driver: str

    @property
    def vram_gib(self) -> float:
        return self.vram_mib / 1024.0


SMI_QUERY = "name,memory.total,compute_cap,driver_version"


def parse_gpu_csv(text: str) -> list[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv``."""
    rows = list(csv.reader(io.StringIO((text or "").strip()), skipinitialspace=True))
    if rows and rows[0] and rows[0][0].strip().lower() == "name":
        rows = rows[1:]
    out: list[GpuInfo] = []
    for row in rows:
        if len(row) < 4:
            continue
        name, mem, cc, drv = (c.strip() for c in row[:4])
        m_mem = re.search(r"\d+", mem)
        m_cc = re.fullmatch(r"(\d+)\.(\d+)", cc)
        if not m_mem or not m_cc:
            raise SetupError(
                f"cannot read compute capability from nvidia-smi for '{name}' (got '{cc}'): the driver is too old to report it. "
                "Update the NVIDIA driver, then run this script again.")
        out.append(GpuInfo(name, int(m_mem.group()), (int(m_cc.group(1)), int(m_cc.group(2))), drv))
    return out


def parse_cuda_header(text: str) -> str | None:
    """The 'CUDA Version: X.Y' of the plain ``nvidia-smi`` header: the highest CUDA the DRIVER supports."""
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", text or "")
    return m.group(1) if m else None


def gpu_generation(cc: tuple[int, int]) -> str:
    """Name the generation from compute capability, never from a marketing name."""
    major, minor = cc
    if major == 12:
        return f"Blackwell, workstation/consumer class (sm_{major}{minor})"
    if major in (10, 11):
        return f"Blackwell-class data-center/embedded (sm_{major}{minor})"
    if major == 9:
        return "Hopper"
    if major == 8:
        return "Ada Lovelace" if minor == 9 else "Ampere"
    if major == 7:
        return "Turing" if minor == 5 else "Volta"
    return f"older than Volta (sm_{major}{minor})"


# --------------------------------------------------------------------------------------------------------------------
# Torch wheel index
# --------------------------------------------------------------------------------------------------------------------
# CUDA builds published per torch minor. 2.12 / 2.13: pytorch.org "previous versions" lists cu126, cu130, cu132 (no cu128, no cu129).
# 2.11: cu128 is listed there; cu130 became the PyPI default in 2.11 (PyTorch dev-discuss announcement). Any other torch minor is
# unknown to this table, so it needs an explicit --cuda-index instead of a guess.
TORCH_BUILDS: dict[tuple[int, int], tuple[str, ...]] = {
    (2, 13): ("cu126", "cu130", "cu132"),
    (2, 12): ("cu126", "cu130", "cu132"),
    (2, 11): ("cu128", "cu130"),
}
EXPERIMENTAL_BUILDS = frozenset({"cu132"})  # never chosen automatically
# Minimum Linux driver per build. cu126: CUDA 12.x minor-version compatibility (>= 525). cu128: CUDA 12.8 GA. cu130: CUDA 13.0 GA.
MIN_DRIVER: dict[str, tuple[int, ...]] = {"cu126": (525,), "cu128": (570, 117), "cu130": (580, 65, 6)}
INDEX_URL = "https://download.pytorch.org/whl/{index}"


def cuda_number(index: str) -> int:
    """``'cu130'`` -> 130."""
    m = re.fullmatch(r"cu(\d{3})", index or "")
    if not m:
        raise SetupError(f"'{index}' is not a CUDA build tag like cu126 or cu130")
    return int(m.group(1))


def min_cuda_for_cc(cc: tuple[int, int]) -> int:
    """Blackwell (sm_100 and up) needs a CUDA >= 12.8 build; nvcc 12.8 is the first that targets sm_100/101/120."""
    return 128 if cc[0] >= 10 else 0


@dataclass(frozen=True)
class TorchPlan:
    index: str
    reason: str
    warnings: tuple[str, ...] = ()

    @property
    def index_url(self) -> str:
        return INDEX_URL.format(index=self.index)


def _fmt_driver(t: tuple[int, ...]) -> str:
    return ".".join(f"{x:02d}" if i == 2 else str(x) for i, x in enumerate(t))


def select_torch_index(driver: str, cc: tuple[int, int], torch_version: str, override: str | None = None) -> TorchPlan:
    """Pick the CUDA build of torch: the highest STABLE build that (a) exists for this torch, (b) can run on this GPU generation and
    (c) is supported by the installed driver. No silent downgrade: if nothing fits, say what to change."""
    tv = version_tuple(torch_version)
    if len(tv) < 2:
        raise SetupError(f"cannot read a torch version from '{torch_version}'")
    minor = (tv[0], tv[1])
    drv = version_tuple(driver)
    if not drv:
        raise SetupError(f"cannot read the driver version from '{driver}'")
    need_cuda = min_cuda_for_cc(cc)
    gen = gpu_generation(cc)
    known = TORCH_BUILDS.get(minor)

    if override:
        n = cuda_number(override)
        warns: list[str] = []
        if known is not None and override not in known:
            raise SetupError(f"torch {torch_version} has no '{override}' build (pytorch.org lists {', '.join(known)}).")
        if n < need_cuda:
            raise SetupError(f"{gen} needs a CUDA >= 12.8 build of torch; '{override}' cannot run on it.")
        if override in MIN_DRIVER:
            if drv < MIN_DRIVER[override]:
                raise SetupError(f"{override} needs driver >= {_fmt_driver(MIN_DRIVER[override])}; this machine has {driver}.")
        else:
            warns.append(f"{override}: the minimum driver is not in this script's table; the torch check at the end decides.")
        if override in EXPERIMENTAL_BUILDS:
            warns.append(f"{override} is experimental in PyTorch.")
        return TorchPlan(override, f"chosen by --cuda-index ({gen}, driver {driver})", tuple(warns))

    if known is None:
        raise SetupError(
            f"torch {torch_version} is not in this script's table of CUDA builds (known minors: "
            f"{', '.join('%d.%d' % k for k in sorted(TORCH_BUILDS))}). Pass --cuda-index cuNNN after checking pytorch.org.")
    stable = [b for b in known if b not in EXPERIMENTAL_BUILDS]
    arch_ok = [b for b in stable if cuda_number(b) >= need_cuda]
    fits = [b for b in arch_ok if b in MIN_DRIVER and drv >= MIN_DRIVER[b]]
    if fits:
        best = max(fits, key=cuda_number)
        return TorchPlan(best, f"highest stable build for torch {torch_version} that {gen} and driver {driver} support "
                               f"(candidates {', '.join(stable)})")
    if arch_ok:
        floor = min((MIN_DRIVER[b] for b in arch_ok if b in MIN_DRIVER), default=None)
        raise SetupError(
            f"{gen} needs torch built with CUDA >= 12.8, and torch {torch_version} offers {', '.join(arch_ok)} for that; "
            f"the lowest of those needs driver >= {_fmt_driver(floor) if floor else '?'} but this machine has {driver}. "
            "Update the NVIDIA driver (a system change this script does not make). Alternatively pass --torch/--torchvision/--cuda-index "
            "for an older torch that still has a cu128 build (pytorch.org lists torch 2.11.0 with torchvision 0.26.0); that departs "
            "from the torch version the repo's requirements.txt names, so it is your call.")
    raise SetupError(f"no CUDA build of torch {torch_version} can run on {gen}; pass --cuda-index after checking pytorch.org.")


def parse_torch_pins(requirements_text: str) -> dict[str, str]:
    """requirements.txt keeps torch/torchvision commented out ON PURPOSE; the commented lines still name the intended versions."""
    pins: dict[str, str] = {}
    for line in (requirements_text or "").splitlines():
        m = re.match(r"^\s*#\s*(torch|torchvision)\s*==\s*([0-9][^\s#]*)", line)
        if m and m.group(1) not in pins:
            pins[m.group(1)] = m.group(2)
    return pins


def commented_pin(requirements_text: str, package: str) -> str | None:
    """Version from a commented ``pip install <package>==X`` hint such as the causal-conv1d line of requirements-gpu.txt."""
    m = re.search(rf"{re.escape(package)}\s*==\s*([0-9][^\s#]*)", requirements_text or "")
    return m.group(1) if m else None


def commented_wheel_url(requirements_text: str, package: str = "flash_attn") -> str | None:
    """A wheel URL from a commented ``pip install --no-deps "<url>"`` line (requirements-gpu.txt keeps one for flash-attn)."""
    for line in (requirements_text or "").splitlines():
        if line.lstrip().startswith("#") and package in line:
            m = re.search(r"https?://[^\s\"']+\.whl", line)
            if m:
                return m.group(0)
    return None


# --------------------------------------------------------------------------------------------------------------------
# Kernels and optional accelerators, chosen from compute capability
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AttentionPlan:
    generation: str
    try_flash_attn2: bool
    notes: tuple[str, ...]


def attention_plan(cc: tuple[int, int]) -> AttentionPlan:
    """FlashAttention-2: Ampere, Ada, Hopper. FlashAttention-3 (Hopper only) is never installed by this script. Blackwell, Volta,
    Turing and unknown generations: PyTorch SDPA."""
    gen = gpu_generation(cc)
    major = cc[0]
    if major == 9:
        return AttentionPlan(gen, True, ("FlashAttention-2 applies (Ampere, Ada, Hopper).",
                                         "FlashAttention-3 is Hopper-only and a beta; this script does not install it."))
    if major == 8:
        return AttentionPlan(gen, True, ("FlashAttention-2 applies (Ampere, Ada, Hopper).",
                                         "FlashAttention-3 is Hopper-only: it does not run on this GPU."))
    if major >= 10:
        return AttentionPlan(gen, False, (
            "flash-attn does not list Blackwell; FlashAttention-3 is Hopper-only (its kernels fail on sm_120 per upstream issue 1987) "
            "and FlashAttention-4 targets B200-class parts; no prebuilt wheel for sm_120 was found.",
            "PyTorch SDPA is used. The checks at the end report which SDPA backends really run here."))
    return AttentionPlan(gen, False, ("FlashAttention-2 needs Ampere or newer; PyTorch SDPA is used.",))


def attention_outcome(cc_major: int, flash_attn_imports: bool) -> str:
    """What ``scripts/train_sft.py`` will pick on its own (lines 716-726): flash_attention_2 when this is a CUDA device with
    capability major >= 8 AND ``import flash_attn`` works, else sdpa. Nothing in the bootstrap sets it."""
    return "flash_attention_2" if (cc_major >= 8 and flash_attn_imports) else "sdpa"


def arch_supported(arch_list: Sequence[str], cc: tuple[int, int]) -> bool:
    """Does this torch build contain code for the device? A cubin ``sm_XY`` runs on the same major with minor >= Y; a PTX
    ``compute_XY`` entry is JIT-compiled for any device with capability >= XY."""
    dev = cc[0] * 10 + cc[1]
    for a in arch_list:
        m = re.fullmatch(r"sm_(\d+)[a-z]?", a)
        if m and int(m.group(1)) // 10 == cc[0] and int(m.group(1)) <= dev:
            return True
        m = re.fullmatch(r"compute_(\d+)[a-z]?", a)
        if m and int(m.group(1)) <= dev:
            return True
    return False


@dataclass(frozen=True)
class EnvTags:
    torch_version: str
    cuda_major: int
    py_tag: str
    abi: str  # 'TRUE' | 'FALSE'

    @property
    def torch_mm(self) -> str:
        v = version_tuple(self.torch_version)
        return f"{v[0]}.{v[1]}"


_FA_WHEEL = re.compile(
    r"^flash_attn-(?P<ver>[^+]+)\+cu(?P<cu>\d+)torch(?P<torch>[0-9.]+)cxx11abi(?P<abi>TRUE|FALSE)-(?P<py>cp\d+)-(?P=py)-linux_x86_64\.whl$")


def pick_flash_attn_asset(names: Sequence[str], tags: EnvTags) -> str | None:
    """Pick the official prebuilt flash-attn wheel whose CUDA major, torch major.minor, ABI and Python tag all match; highest version."""
    best: tuple[tuple[int, ...], str] | None = None
    for n in names:
        m = _FA_WHEEL.match(n)
        if not m:
            continue
        if (int(m["cu"]), m["torch"], m["abi"], m["py"]) != (tags.cuda_major, tags.torch_mm, tags.abi, tags.py_tag):
            continue
        key = version_tuple(m["ver"])
        if best is None or key > best[0]:
            best = (key, n)
    return best[1] if best else None


def causal_conv1d_wheel_url(version: str, tags: EnvTags) -> tuple[str, str]:
    """The official wheel URL, built exactly as causal-conv1d's own setup.py builds it (read from the 1.7.0 sdist)."""
    if tags.cuda_major not in (11, 12, 13):
        raise SetupError(f"causal-conv1d publishes wheels for CUDA 11, 12 and 13 only, not {tags.cuda_major}")
    fname = f"causal_conv1d-{version}+cu{tags.cuda_major}torch{tags.torch_mm}cxx11abi{tags.abi}-{tags.py_tag}-{tags.py_tag}-linux_x86_64.whl"
    return f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v{version}/{fname}", fname


# --------------------------------------------------------------------------------------------------------------------
# .env handling (values are never returned in anything that gets printed)
# --------------------------------------------------------------------------------------------------------------------
TOKEN_KEYS = ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY", "NVIDIA_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
              "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
MINIMAL_ENV = "HF_TOKEN=your_huggingface_token_here\n"


def is_placeholder(value: str | None) -> bool:
    """Same rule as ``dental_agent.utils.inputs.hub_token``: empty or ``your_...`` is not a token; ``/path/to`` is not a path."""
    v = (value or "").strip().strip("'\"")
    return (not v) or v.lower().startswith("your_") or v.startswith("/path/to")


def read_env_value(text: str, key: str) -> str | None:
    """Last assignment wins, like python-dotenv. Understands ``export``, quotes and trailing `` # comment``."""
    found: str | None = None
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=\s*(.*?)\s*$")
    for line in (text or "").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = pat.match(line)
        if not m:
            continue
        raw = m.group(1)
        if raw[:1] in ("'", '"'):
            end = raw.find(raw[0], 1)
            found = raw[1:end] if end > 0 else raw[1:]
        else:
            found = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
    return found


def set_env_value(text: str, key: str, value: str) -> str:
    """Replace the last active assignment of ``key`` (or append one). Every other line is kept byte for byte."""
    lines = (text or "").splitlines(keepends=True)
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=")
    idx = [i for i, ln in enumerate(lines) if not ln.lstrip().startswith("#") and pat.match(ln)]
    new = f"{key}={value}" if not re.search(r"[\s#]", value) else f'{key}="{value}"'
    if idx:
        i = idx[-1]
        lines[i] = new + ("\n" if lines[i].endswith("\n") else "")
        return "".join(lines)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return "".join(lines) + new + "\n"


def env_status(text: str, key: str) -> str:
    v = read_env_value(text, key)
    if v is None:
        return "missing"
    return "placeholder" if is_placeholder(v) else "set"


def git_env_state(repo: str | Path) -> tuple[bool | None, bool | None]:
    """``(.env is git-ignored, .env is tracked)``; ``None`` when git cannot say."""
    def rc(args: list[str]) -> int | None:
        try:
            return subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        except OSError:
            return None
    ign = rc(["check-ignore", "-q", ".env"])
    trk = rc(["ls-files", "--error-unmatch", ".env"])
    return (None if ign not in (0, 1) else ign == 0), (None if trk not in (0, 1) else trk == 0)


@dataclass
class EnvResult:
    path: str
    created: bool = False
    filled_hf_token: bool = False
    chmod_fixed: bool = False
    mode: str = ""
    git_ignored: bool | None = None
    git_tracked: bool | None = None
    statuses: dict[str, str] = field(default_factory=dict)  # key -> set | placeholder | missing  (never a value)
    hf_token_source: str = "none"  # '.env' | 'environment' | 'none'


def _usable_token(environ: Mapping[str, str]) -> str | None:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = environ.get(k)
        if v and not is_placeholder(v):
            return v
    return None


def ensure_env_file(repo: str | Path, environ: Mapping[str, str] | None = None,
                    git_state: Callable[[str | Path], tuple[bool | None, bool | None]] = git_env_state) -> EnvResult:
    """Create ``.env`` from ``.env.example`` when absent (mode 600 from the first byte, created exclusively). An existing ``.env`` keeps
    its content; only group/other permission bits are removed, and only on a file this user owns."""
    repo = Path(repo)
    environ = os.environ if environ is None else environ
    env_path, template = repo / ".env", repo / ".env.example"
    res = EnvResult(path=str(env_path))
    if not env_path.exists():
        text = template.read_text(encoding="utf-8") if template.is_file() else MINIMAL_ENV
        tok = _usable_token(environ)
        if tok:
            text = set_env_value(text, "HF_TOKEN", tok)
            res.filled_hf_token = True
        fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.chmod(env_path, 0o600)
        res.created = True
    else:
        st = env_path.stat()
        mine = not hasattr(os, "getuid") or st.st_uid == os.getuid()
        if stat.S_IMODE(st.st_mode) & 0o077 and mine:
            os.chmod(env_path, 0o600)
            res.chmod_fixed = True
    res.mode = oct(stat.S_IMODE(env_path.stat().st_mode))
    body = env_path.read_text(encoding="utf-8", errors="replace")
    res.statuses = {k: env_status(body, k) for k in TOKEN_KEYS}
    res.git_ignored, res.git_tracked = git_state(repo)
    res.hf_token_source = ".env" if res.statuses.get("HF_TOKEN") == "set" else ("environment" if _usable_token(environ) else "none")
    return res


def check_hf_token(token: str) -> tuple[str, str]:
    """One cheap ``whoami`` call. Returns ``(status, detail)``; the detail never contains the token."""
    try:
        from huggingface_hub import HfApi
    except Exception:  # noqa: BLE001
        return "unavailable", "huggingface_hub is not importable yet"
    try:
        info = HfApi().whoami(token=token)
    except Exception as e:  # noqa: BLE001
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            return "rejected", f"HTTP {code} (invalid token, or the Hub is blocked from this network)"
        return "unreachable", type(e).__name__
    return "ok", str(info.get("name", "?"))


# --------------------------------------------------------------------------------------------------------------------
# The command handed to `train_sft.py --preflight`
# --------------------------------------------------------------------------------------------------------------------
# DISCOVERY inputs only, mirroring what the SFT notebook's GPU path passes (notebooks/VLM_Dental_Colab_SFT.ipynb, section 6) so the
# doctor looks for the same traces, images, dev split and manifest. They are not training settings and this module never adds
# --precision, --max-seq-len, LoRA or any hyperparameter: train_sft.py's own defaults apply to everything not listed. The notebook's
# own preflight (it re-runs its exact command with --preflight) stays the authoritative one. A user flag always wins; a default is
# dropped when the user already gave that flag or its --no- twin, so no flag is ever passed twice.
_PREFLIGHT_DEFAULTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("--track",), ("--track", "{track}")),
    (("--stage",), ("--stage", "{stage}")),
    (("--data-dir",), ("--data-dir", "data")),
    (("--dev-split",), ("--dev-split", "data/splits/dentex_dev.json")),
    (("--canonical-resize", "--no-canonical-resize"), ("--canonical-resize",)),
    (("--canonical-letterbox", "--no-canonical-letterbox"), ("--no-canonical-letterbox",)),
)


def _has_flag(args: Sequence[str], names: Sequence[str]) -> bool:
    return any(a == n or a.startswith(n + "=") for a in args for n in names)


def build_preflight_argv(stage: str, track: str, model_id: str | None, hf_repo: str | None, user_args: Sequence[str] = ()) -> list[str]:
    """Arguments after ``python`` for the preflight (script path first, ``--preflight`` last)."""
    user = list(user_args)
    out: list[str] = ["scripts/train_sft.py"]
    for names, argv in _PREFLIGHT_DEFAULTS:
        if not _has_flag(user, names):
            out += [a.format(track=track, stage=stage) for a in argv]
    if model_id and not _has_flag(user, ("--model-id",)):
        out += ["--model-id", model_id]
    if hf_repo and not _has_flag(user, ("--hf-repo",)):
        out += ["--hf-repo", hf_repo]
    return out + user + ["--preflight"]


# --------------------------------------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------------------------------------
# GB still to be written on a blank machine. model: docs/RUNBOOK_SFT_GRPO.md says 19 GB. The other numbers are ALLOWANCES chosen
# here, not measurements (the sandbox that wrote them reaches neither the PyTorch index nor the Hub): override with --min-free-gb.
DISK_ALLOWANCE_GB = {"python_env": 10.0, "wheelhouse": 6.0, "model": 19.0, "datasets": 10.0, "headroom": 5.0}


@dataclass(frozen=True)
class DiskVerdict:
    ok: bool
    need_repo_gb: float
    need_home_gb: float
    messages: tuple[str, ...]


def disk_plan(repo_free_gb: float, home_free_gb: float, same_fs: bool, have_gb: Mapping[str, float],
              min_free_gb: float | None = None) -> DiskVerdict:
    """What is still missing counts, not the totals: a re-run after a stalled download must not refuse because the half-finished copy
    is already on disk."""
    left = {k: max(0.0, DISK_ALLOWANCE_GB[k] - have_gb.get(k, 0.0)) for k in ("python_env", "wheelhouse", "model", "datasets")}
    need_repo = left["python_env"] + left["model"] + left["datasets"] + DISK_ALLOWANCE_GB["headroom"]
    need_home = left["wheelhouse"] + 1.0  # 1 GB: the uv-managed Python
    if min_free_gb is not None:
        need_repo = float(min_free_gb)
    msgs: list[str] = []
    if same_fs:
        need_repo += need_home
        need_home = 0.0
        if repo_free_gb < need_repo:
            msgs.append(f"{repo_free_gb:.0f} GB free where the repo and ~ live (same filesystem); about {need_repo:.0f} GB still needed")
    else:
        if repo_free_gb < need_repo:
            msgs.append(f"{repo_free_gb:.0f} GB free at the repo; about {need_repo:.0f} GB still needed")
        if home_free_gb < need_home:
            msgs.append(f"{home_free_gb:.0f} GB free at ~ (wheelhouse, uv Python); about {need_home:.0f} GB still needed")
    return DiskVerdict(not msgs, need_repo, need_home, tuple(msgs))


def dir_size_gb(path: str | Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / 2**30


# --------------------------------------------------------------------------------------------------------------------
# pip constraints
# --------------------------------------------------------------------------------------------------------------------
def constraints_text(torch_version: str, torchvision_version: str) -> str:
    return ("# Written by scripts/setup_gpu_server.sh: the installed torch build, so that no later pip command can replace it.\n"
            f"torch=={torch_version}\ntorchvision=={torchvision_version}\n")


def pip_conf_text(constraints_path: str | Path) -> str:
    """Site config of the venv: ``pip install`` / ``pip download`` run from the notebook (``!pip install -U ...``) honour the pin too."""
    p = str(constraints_path)
    return f"[install]\nconstraint = {p}\n\n[download]\nconstraint = {p}\n"


# --------------------------------------------------------------------------------------------------------------------
# State file and summary
# --------------------------------------------------------------------------------------------------------------------
def load_state(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record(path: str | Path, key: str, value: Any) -> None:
    st = load_state(path)
    st[key] = value
    Path(path).write_text(json.dumps(st, indent=1, sort_keys=True), encoding="utf-8")


_SUMMARY_ORDER = (
    ("machine", "Machine"), ("torch", "torch / CUDA build"), ("torch_check", "torch check"), ("packages", "Python packages"),
    ("env_file", ".env"), ("hf_token", "HF token"), ("project", "Project"), ("bitsandbytes", "bitsandbytes (QLoRA)"),
    ("sdpa", "PyTorch SDPA backends"), ("flash_attn", "flash-attn"), ("attention_backend", "Attention backend train_sft.py will pick"),
    ("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d"), ("kernelspec", "Jupyter kernel"),
    ("model", "Base model"), ("data", "Traces / images"), ("preflight", "SFT preflight"), ("precision_hint", "Precision hint"),
)


def format_summary(state: Mapping[str, Any]) -> str:
    lines = ["", "=" * 78, "SUMMARY", "=" * 78]
    shown = set()
    for key, label in _SUMMARY_ORDER:
        if key in state:
            lines.append(f"{label + ':':<42}{state[key]}")
            shown.add(key)
    missing = state.get("missing") or []
    lines.append("")
    lines.append("Missing / not done: " + ("nothing" if not missing else ""))
    lines.extend(f"  - {m}" for m in missing)
    if state.get("next_step"):
        lines += ["", "NEXT STEP:", f"  {state['next_step']}"]
    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Machine report
# --------------------------------------------------------------------------------------------------------------------
def run_cmd(argv: Sequence[str], timeout: int = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr if p.returncode else "")
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""


def _ram_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    return None


def gpu_memory_in_use(runner: Callable[[Sequence[str]], tuple[int, str]] = run_cmd) -> list[str]:
    """Read-only look at who already holds GPU memory (shared servers). The script never touches those processes."""
    rc, out = runner(["nvidia-smi", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader"])
    return [ln.strip() for ln in out.splitlines() if ln.strip()] if rc == 0 else []


def collect_report(repo: Path, venv: Path, wheelhouse: Path, min_free_gb: float | None,
                   runner: Callable[[Sequence[str]], tuple[int, str]] = run_cmd) -> tuple[dict[str, Any], list[str]]:
    """Returns ``(info, printable lines)``. ``info['status']`` is ok | no_gpu | disk."""
    lines = ["== Machine report =="]
    rc, csv_out = runner(["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv"])
    if rc != 0:
        why = "nvidia-smi was not found" if rc == 127 else f"nvidia-smi failed: {csv_out.strip().splitlines()[0] if csv_out.strip() else rc}"
        lines += [f"No usable NVIDIA GPU ({why}).",
                  "This script is for Linux servers with an NVIDIA driver; install the driver first (the TPU path is notebook-only)."]
        return {"status": "no_gpu"}, lines
    gpus = parse_gpu_csv(csv_out)
    if not gpus:
        return {"status": "no_gpu"}, lines + ["nvidia-smi listed no GPU."]
    _rc, header = runner(["nvidia-smi"])
    cuda_hdr = parse_cuda_header(header)
    g0 = gpus[0]
    for i, g in enumerate(gpus):
        lines.append(f"GPU {i}: {g.name} | {g.vram_gib:.1f} GiB | compute capability {g.cc[0]}.{g.cc[1]} ({gpu_generation(g.cc)})")
    if len({(g.cc, g.driver) for g in gpus}) > 1:
        lines.append("WARNING: the GPUs differ; every decision below uses GPU 0.")
    lines.append(f"Driver: {g0.driver} | highest CUDA the driver supports (nvidia-smi header): {cuda_hdr or 'unknown'}")
    busy = gpu_memory_in_use(runner)
    if busy:
        lines.append(f"GPU memory is already held by {len(busy)} process(es) (pid, memory, name); this script never touches them:")
        lines += [f"    {b}" for b in busy[:8]]
    cores, ram = os.cpu_count(), _ram_gib()
    lines.append(f"CPU cores: {cores} | RAM: {ram:.1f} GiB" if ram else f"CPU cores: {cores} | RAM: unknown")

    home = Path.home()
    probe = repo
    repo_free = shutil.disk_usage(probe).free / 2**30
    home_free = shutil.disk_usage(home).free / 2**30
    same_fs = os.stat(probe).st_dev == os.stat(home).st_dev
    have = {"python_env": dir_size_gb(venv) if venv.exists() else 0.0,
            "wheelhouse": dir_size_gb(wheelhouse) if wheelhouse.exists() else 0.0,
            "model": dir_size_gb(repo / "data/models/Qwen3.5-9B") if (repo / "data/models/Qwen3.5-9B").exists() else 0.0,
            "datasets": sum(dir_size_gb(repo / d) for d in ("data/dentex", "data/Tufts") if (repo / d).exists())}
    verdict = disk_plan(repo_free, home_free, same_fs, have, min_free_gb)
    lines.append(f"Disk free: {repo_free:.0f} GB at the repo, {home_free:.0f} GB at ~" + (" (same filesystem)" if same_fs else ""))
    done = ", ".join(f"{k} {v:.1f} GB" for k, v in have.items() if v > 0.05)
    lines.append("Already on disk (counted as done): " + (done or "nothing"))
    lines.append(f"Still needed: about {verdict.need_repo_gb + verdict.need_home_gb:.0f} GB (model 19 GB is from the repo docs; the other "
                 "figures are allowances, not measurements; --min-free-gb overrides)")
    vram_left = g0.vram_gib - 18.0
    hint = (f"hint only, nothing is changed (PRECISION is set in the notebook): the 9B weights in 16-bit are about 18 GB, which leaves "
            f"about {max(vram_left, 0):.0f} GiB of this {g0.vram_gib:.0f} GiB card for activations and optimizer state; "
            "no memory peak has been measured by this script")
    lines.append("Precision " + hint + ".")
    lines.append("If space is tight on ext4 the reserved-blocks share (default 5%) can be lowered by the machine's owner with "
                 "`sudo tune2fs -m 1 <device>`; this script never runs it.")
    info = {
        "status": "ok" if verdict.ok else "disk", "gpu_name": g0.name, "vram_gib": round(g0.vram_gib, 1), "cc": list(g0.cc),
        "generation": gpu_generation(g0.cc), "driver": g0.driver, "cuda_header": cuda_hdr, "cores": cores, "ram_gib": ram,
        "repo_free_gb": round(repo_free, 1), "home_free_gb": round(home_free, 1), "disk_messages": list(verdict.messages),
        "precision_hint": hint,
    }
    if not verdict.ok:
        lines += ["NOT ENOUGH FREE DISK:"] + [f"  - {m}" for m in verdict.messages]
        lines.append("Free space (or point --wheelhouse / the repo at a bigger disk) and run again; everything done so far is kept.")
    return info, lines


# --------------------------------------------------------------------------------------------------------------------
# Commands that run with the venv's Python (torch and the project are imported lazily)
# --------------------------------------------------------------------------------------------------------------------
def _first_line(e: BaseException) -> str:
    s = str(e).strip().splitlines()
    return (s[0] if s else "")[:200]


def env_tags() -> EnvTags:
    import torch
    cuda = torch.version.cuda
    if not cuda:
        raise SetupError("the installed torch is a CPU build (torch.version.cuda is None)")
    return EnvTags(torch.__version__, int(cuda.split(".")[0]), f"cp{sys.version_info.major}{sys.version_info.minor}",
                   str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())


def verify_torch() -> dict[str, Any]:
    import torch
    out: dict[str, Any] = {"torch": torch.__version__, "cuda_build": torch.version.cuda}
    if not torch.cuda.is_available():
        out["ok"] = False
        out["detail"] = "torch.cuda.is_available() is False"
        return out
    cc = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    out.update(device=torch.cuda.get_device_name(0), cc=f"{cc[0]}.{cc[1]}", arch_list=arch, arch_ok=arch_supported(arch, cc),
               bf16=bool(torch.cuda.is_bf16_supported()))
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        out["matmul"] = bool(torch.isfinite(a @ a).all().item())
    except Exception as e:  # noqa: BLE001
        out["matmul"] = False
        out["detail"] = f"{type(e).__name__}: {_first_line(e)}"
    out["ok"] = bool(out["arch_ok"] and out.get("matmul"))
    return out


def verify_attention() -> dict[str, Any]:
    """Real forward + backward passes: every PyTorch SDPA backend on its own, and flash_attn when it is installed."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    res: dict[str, Any] = {}
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        be = getattr(SDPBackend, name, None)
        if be is None:
            continue
        try:
            q, k, v = (torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
            with sdpa_kernel(be):
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            o.float().sum().backward()
            torch.cuda.synchronize()
            res[name] = "ok" if bool(torch.isfinite(o).all() and torch.isfinite(q.grad).all()) else "non-finite output"
        except Exception as e:  # noqa: BLE001
            res[name] = f"fail: {type(e).__name__}: {_first_line(e)}"
    try:
        from flash_attn import flash_attn_func
    except ModuleNotFoundError:
        res["flash_attn"] = "not installed"
    except Exception as e:  # noqa: BLE001
        res["flash_attn"] = f"import failed: {type(e).__name__}: {_first_line(e)}"
    else:
        try:
            q, k, v = (torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
            o = flash_attn_func(q, k, v, causal=True)
            o.float().sum().backward()
            torch.cuda.synchronize()
            res["flash_attn"] = "ok" if bool(torch.isfinite(o).all() and torch.isfinite(q.grad).all()) else "non-finite output"
        except Exception as e:  # noqa: BLE001
            res["flash_attn"] = f"fail: {type(e).__name__}: {_first_line(e)}"
    return res


def verify_fla() -> dict[str, Any]:
    """Resolve the kernel the way transformers 5.19 does (import ``fla``, then ``fla.ops.gated_delta_rule``): any exception there makes
    transformers fall back to its reference PyTorch path with a misleading 'not installed' warning, so the real error is shown here.
    Then one forward + backward with ``use_qk_l2norm_in_kernel=True``, the call Qwen3.5 makes."""
    import importlib
    import torch
    try:
        importlib.import_module("fla")
        fn = importlib.import_module("fla.ops.gated_delta_rule").chunk_gated_delta_rule
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"import failed: {type(e).__name__}: {_first_line(e)}"}
    try:
        import torch.nn.functional as F
        B, T, H, HV, K = 1, 512, 2, 4, 128
        dev, dt = "cuda", torch.bfloat16
        q = torch.randn(B, T, H, K, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(B, T, H, K, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(B, T, HV, K, device=dev, dtype=dt, requires_grad=True)
        g = F.logsigmoid(torch.rand(B, T, HV, device=dev, dtype=dt))
        beta = torch.rand(B, T, HV, device=dev, dtype=dt).sigmoid()
        torch.cuda.reset_peak_memory_stats()
        o, _ = fn(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        o.float().sum().backward()
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(o).all() and torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all())
        return {"ok": ok, "detail": f"forward+backward on a toy shape (T={T}); peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
                                    if ok else "non-finite values"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"kernel call failed: {type(e).__name__}: {_first_line(e)}"}


def verify_causal_conv1d() -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    try:
        from causal_conv1d import causal_conv1d_fn
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "installed": not isinstance(e, ModuleNotFoundError), "detail": f"import failed: {type(e).__name__}: {_first_line(e)}"}
    try:
        x = torch.randn(1, 64, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
        got = causal_conv1d_fn(x, w, None, activation="silu")
        ref = F.silu(F.conv1d(x, w.unsqueeze(1), padding=3, groups=64)[:, :, :256])
        err = float((got.float() - ref.float()).abs().max())
        return {"ok": err < 5e-2, "installed": True, "detail": f"max abs difference to the PyTorch reference {err:.4f}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "installed": True, "detail": f"kernel call failed: {type(e).__name__}: {_first_line(e)}"}


def verify_bnb() -> dict[str, Any]:
    import torch
    try:
        import bitsandbytes as bnb
        layer = bnb.nn.Linear4bit(128, 128, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
        y = layer(torch.randn(2, 128, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
        return {"ok": bool(torch.isfinite(y).all()), "detail": f"bitsandbytes {bnb.__version__}: 4-bit NF4 layer ran a forward pass"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(e).__name__}: {_first_line(e)}"}


def http_status(url: str, method: str = "HEAD", timeout: int = 20) -> int:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "vlm-dental-setup"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # URLs are fixed https ones
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return 0


def github_release_assets(repo: str, releases: int = 4, timeout: int = 25) -> dict[str, str]:
    """``{asset file name: download URL}`` of the newest releases (unauthenticated API call: no token of the user's is ever sent).
    The download URL comes from the API because a release tag is not always the version in the file name."""
    assets: dict[str, str] = {}
    hdr = {"User-Agent": "vlm-dental-setup", "Accept": "application/vnd.github+json"}

    def get(url: str) -> Any:
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    for rel in get(f"https://api.github.com/repos/{repo}/releases?per_page={releases}"):
        for page in range(1, 4):
            batch = get(f"{rel['assets_url']}?per_page=100&page={page}")
            for a in batch:
                assets.setdefault(a["name"], a["browser_download_url"])
            if len(batch) < 100:
                break
    return assets


def fetch_data(repo: Path, stage: str, track: str, want_model: bool, want_data: bool, data_dir: str = "data") -> dict[str, Any]:
    """Uses the code that already exists: ``resolve_qwen_model`` (resumable download into data/models/Qwen3.5-9B), then
    ``resolve_stage_traces`` and ``ensure_trace_images`` (surgical: only what is missing). Nothing is re-implemented."""
    os.chdir(repo)
    from dental_agent.utils import inputs as inp  # importing dental_agent loads .env (dental_agent/__init__.py)
    out: dict[str, Any] = {}
    if want_model:
        path, source = inp.resolve_qwen_model()
        out.update(model_path=path, model_source=source)
    if want_data:
        res = inp.resolve_stage_traces(stage, track, data_dir, hub="download")
        out["trace_files"] = [str(p) for p in res.files]
        chk = inp.ensure_trace_images(res.files, data_dir)  # no exclude: the dev-split images are needed later for the K sweep
        out.update(images=f"{chk.n_unique} unique, {chk.fetched} fetched now")
    return out


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def _emit(obj: Any) -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))


def _opt(*flags: str, **kw: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    return flags, kw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    state = _opt("--state", required=True)
    repo = _opt("--repo", default=str(REPO_ROOT))
    table: dict[str, tuple[str, list[Any]]] = {
        "report": ("inspect the machine, print the report, check disk",
                   [repo, _opt("--venv", required=True), _opt("--wheelhouse", required=True), state, _opt("--min-free-gb", type=float)]),
        "plan": ("choose the torch index and the attention plan from the report",
                 [state, _opt("--torch-version", required=True), _opt("--cuda-index"), _opt("--no-flash-attn", action="store_true")]),
        "pin": ("print one pin read from the commented lines of requirements.txt",
                [_opt("which", choices=["torch", "torchvision"]), _opt("--requirements", required=True)]),
        "get": ("print one value of the state file", [state, _opt("key")]),
        "commented": ("print a version / wheel URL kept in a comment of requirements-gpu.txt",
                      [_opt("what", choices=["causal-conv1d", "flash-wheel"]), _opt("--requirements-gpu", required=True)]),
        "record": ("set one key in the state file", [state, _opt("key"), _opt("value")]),
        "append-missing": ("add one line to the list of missing things", [state, _opt("text")]),
        "summary": ("print the final summary", [state]),
        "env-tags": ("torch / CUDA / Python / ABI tags of the venv (JSON)", []),
        "find-flash-attn-wheel": ("look for an official prebuilt flash-attn wheel matching this torch", [state]),
        "find-causal-conv1d-wheel": ("probe the official causal-conv1d wheel URL for this torch", [_opt("--version", required=True), state]),
        "write-constraints": ("pin the installed torch/torchvision", [_opt("--out", required=True), _opt("--pip-conf", required=True)]),
        "torch-status": ("exit 0 when the installed torch already is the wanted build",
                         [_opt("--torch-version", required=True), _opt("--torchvision-version", required=True), _opt("--index", required=True)]),
        "attn-outcome": ("record which attention backend train_sft.py will pick on its own",
                         [state, _opt("--flash-imports", choices=["true", "false"], required=True)]),
        "ensure-env": ("create / check .env", [repo, state]),
        "check-hf": ("validate the HF token with one whoami call", [repo, state]),
        "verify": ("run a real forward pass of one component",
                   [_opt("what", choices=["torch", "attention", "fla", "causal-conv1d", "bnb"]), state]),
        "fetch-data": ("base model, traces and trace images through dental_agent.utils.inputs",
                       [repo, _opt("--stage", required=True), _opt("--track", required=True), _opt("--no-model", action="store_true"),
                        _opt("--no-data", action="store_true"), state]),
        "preflight-argv": ("print the preflight arguments, one per line",
                           [repo, _opt("--stage", required=True), _opt("--track", required=True), _opt("--model-id"),
                            _opt("user_args", nargs="*")]),
    }
    for name, (help_, opts) in table.items():
        sp = sub.add_parser(name, help=help_)
        for flags, kw in opts:
            sp.add_argument(*flags, **kw)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    try:
        return _dispatch(a)
    except SetupError as e:
        print(f"[SETUP ERROR] {e}", file=sys.stderr)
        return 2


def _dispatch(a: argparse.Namespace) -> int:  # a flat command table
    if a.cmd == "report":
        info, lines = collect_report(Path(a.repo), Path(a.venv), Path(a.wheelhouse), a.min_free_gb)
        print("\n".join(lines))
        for k, v in info.items():
            record(a.state, f"report_{k}", v)
        if info["status"] == "ok":
            record(a.state, "machine", f"{info['gpu_name']}, {info['vram_gib']} GiB, sm_{info['cc'][0]}{info['cc'][1]} "
                                       f"({info['generation']}), driver {info['driver']}")
            record(a.state, "precision_hint", info["precision_hint"])
        return {"ok": 0, "no_gpu": 2, "disk": 3}[info["status"]]

    if a.cmd == "plan":
        st = load_state(a.state)
        cc, drv = tuple(st["report_cc"]), st["report_driver"]
        tp = select_torch_index(drv, cc, a.torch_version, a.cuda_index)
        ap = attention_plan(cc)
        print(f"torch {a.torch_version}: CUDA build {tp.index} ({tp.reason})")
        for w in tp.warnings:
            print(f"  note: {w}")
        print(f"GPU generation from compute capability: {ap.generation}")
        for n in ap.notes:
            print(f"  {n}")
        want_fa = ap.try_flash_attn2 and not a.no_flash_attn
        if a.no_flash_attn and ap.try_flash_attn2:
            print("  --no-flash-attn: flash-attn is not installed by this run.")
        record(a.state, "torch_index", tp.index)
        record(a.state, "torch_reason", tp.reason)
        record(a.state, "plan_flash_attn2", want_fa)
        record(a.state, "plan_cc_major", cc[0])
        record(a.state, "torch", f"{a.torch_version} from {tp.index_url} ({tp.reason})")
        return 0

    if a.cmd == "pin":
        pins = parse_torch_pins(Path(a.requirements).read_text(encoding="utf-8"))
        if a.which not in pins:
            raise SetupError(f"requirements.txt has no commented '# {a.which}==X' line; pass --torch and --torchvision")
        print(pins[a.which])
        return 0

    if a.cmd == "get":
        v = load_state(a.state).get(a.key, "")
        print(json.dumps(v) if isinstance(v, bool) else v)
        return 0

    if a.cmd == "commented":
        text = Path(a.requirements_gpu).read_text(encoding="utf-8")
        v = commented_pin(text, "causal-conv1d") if a.what == "causal-conv1d" else commented_wheel_url(text)
        print(v or "")
        return 0 if v else 1

    if a.cmd == "record":
        record(a.state, a.key, a.value)
        return 0

    if a.cmd == "append-missing":
        st = load_state(a.state)
        record(a.state, "missing", [*(st.get("missing") or []), a.text])
        return 0

    if a.cmd == "summary":
        print(format_summary(load_state(a.state)))
        return 0

    if a.cmd == "env-tags":
        t = env_tags()
        _emit(dict(torch_version=t.torch_version, cuda_major=t.cuda_major, py_tag=t.py_tag, abi=t.abi, torch_mm=t.torch_mm))
        return 0

    if a.cmd == "find-flash-attn-wheel":
        t = env_tags()
        try:
            assets = github_release_assets("Dao-AILab/flash-attention")
        except Exception as e:  # noqa: BLE001
            record(a.state, "flash_attn_search", f"release list unreachable ({type(e).__name__})")
            return 1
        hit = pick_flash_attn_asset(list(assets), t)
        record(a.state, "flash_attn_search",
               f"{len(assets)} official release assets checked for cu{t.cuda_major} / torch {t.torch_mm} / {t.py_tag} / abi {t.abi}: "
               + ("match found" if hit else "no match"))
        if hit:
            print(assets[hit])
        return 0 if hit else 1

    if a.cmd == "find-causal-conv1d-wheel":
        t = env_tags()
        url, _fname = causal_conv1d_wheel_url(a.version, t)
        code = http_status(url)
        record(a.state, "causal_conv1d_search", f"{url} -> HTTP {code or 'unreachable'}")
        if code == 200:
            print(url)
            return 0
        return 1

    if a.cmd == "write-constraints":
        from importlib.metadata import version
        Path(a.out).write_text(constraints_text(version("torch"), version("torchvision")), encoding="utf-8")
        Path(a.pip_conf).write_text(pip_conf_text(Path(a.out).resolve()), encoding="utf-8")
        print(f"pinned torch=={version('torch')} torchvision=={version('torchvision')} in {a.out}")
        return 0

    if a.cmd == "torch-status":
        from importlib.metadata import PackageNotFoundError, version
        try:
            ok = (version("torch") == f"{a.torch_version}+{a.index}") and (version("torchvision") == f"{a.torchvision_version}+{a.index}")
        except PackageNotFoundError:
            ok = False
        return 0 if ok else 1

    if a.cmd == "attn-outcome":
        major = int(load_state(a.state)["plan_cc_major"])
        outcome = attention_outcome(major, a.flash_imports == "true")
        warn = (" WARNING: upstream lists no FlashAttention-2 support for Blackwell; uninstall flash-attn if training misbehaves."
                if outcome == "flash_attention_2" and major >= 10 else "")
        record(a.state, "attention_backend", f"{outcome} (picked by scripts/train_sft.py, not by this script: flash_attention_2 only when "
                                              f"capability major >= 8 and `import flash_attn` works){warn}")
        print(f"train_sft.py will use attn_implementation={outcome}")
        return 0

    if a.cmd == "ensure-env":
        r = ensure_env_file(a.repo)
        print(f".env: {'created from .env.example' if r.created else 'already present'} | mode {r.mode}"
              + (" | permissions tightened to 600" if r.chmod_fixed else "") + (" | HF_TOKEN taken from the environment" if r.filled_hf_token else ""))
        print("      git-ignored: " + {True: "yes", False: "NO - add .env to .gitignore before putting a token in it", None: "unknown"}[r.git_ignored]
              + " | tracked by git: " + {True: "YES - remove it from the index", False: "no", None: "unknown"}[r.git_tracked])
        print("      token slots (never the values): " + ", ".join(f"{k}={v}" for k, v in r.statuses.items()))
        record(a.state, "env_file", f"{'created' if r.created else 'existing'}, mode {r.mode}, git-ignored: {r.git_ignored}")
        if r.git_ignored is False or r.git_tracked:
            record(a.state, "missing", [*(load_state(a.state).get("missing") or []), ".env is not safely git-ignored"])
        return 0

    if a.cmd == "check-hf":
        repo = Path(a.repo)
        body = (repo / ".env").read_text(encoding="utf-8", errors="replace") if (repo / ".env").is_file() else ""
        tok = read_env_value(body, "HF_TOKEN")
        if is_placeholder(tok):
            tok = _usable_token(os.environ)
        if not tok:
            print("HF token: none set (.env still holds the placeholder). Put it in .env as HF_TOKEN=..., then run this script again.")
            record(a.state, "hf_token", "not set")
            record(a.state, "missing", [*(load_state(a.state).get("missing") or []), "HF_TOKEN (needed for the model/trace downloads and checkpoint upload)"])
            return 0
        status, detail = check_hf_token(tok)
        print(f"HF token: {status} ({detail})")
        record(a.state, "hf_token", f"{status} ({detail})")
        return 0

    if a.cmd == "verify":
        fn = {"torch": verify_torch, "attention": verify_attention, "fla": verify_fla, "causal-conv1d": verify_causal_conv1d, "bnb": verify_bnb}[a.what]
        res = fn()
        _emit(res)
        if a.what == "torch":
            record(a.state, "torch_check", f"{res.get('torch')} (CUDA {res.get('cuda_build')}) on {res.get('device', 'no GPU')}: "
                   + ("ok, built for this GPU, bf16 matmul ran" if res.get("ok") else f"FAILED {res.get('detail', 'no kernel image for this GPU')}"))
            return 0 if res.get("ok") else 1
        if a.what == "attention":
            record(a.state, "sdpa", ", ".join(f"{k}={v.split(':')[0]}" for k, v in res.items() if k != "flash_attn"))
            record(a.state, "flash_attn_check", res.get("flash_attn", ""))
            return 0
        key = {"fla": "fla", "causal-conv1d": "causal_conv1d", "bnb": "bitsandbytes"}[a.what]
        record(a.state, key, ("ok - " if res["ok"] else "FAILED - ") + res["detail"])
        return 0 if res["ok"] else 1

    if a.cmd == "fetch-data":
        try:
            out = fetch_data(Path(a.repo), a.stage, a.track, not a.no_model, not a.no_data)
        except Exception as e:  # noqa: BLE001 (InputNotFound lists what was searched and the fixing command)
            print(f"[DATA] {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        _emit(out)
        if "model_path" in out:
            record(a.state, "model", f"{out['model_path']} (source: {out['model_source']})")
            record(a.state, "model_path", out["model_path"])
        if "images" in out:
            record(a.state, "data", f"stage {a.stage} / track {a.track}: {len(out['trace_files'])} trace file(s), images {out['images']}")
        return 0

    if a.cmd == "preflight-argv":
        body = (Path(a.repo) / ".env").read_text(encoding="utf-8", errors="replace") if (Path(a.repo) / ".env").is_file() else ""
        hf_repo = read_env_value(body, "HF_ARTIFACT_REPO") or os.environ.get("HF_ARTIFACT_REPO")
        print("\n".join(build_preflight_argv(a.stage, a.track, a.model_id, hf_repo if hf_repo and not is_placeholder(hf_repo) else None,
                                              a.user_args)))
        return 0

    raise SetupError(f"unknown command {a.cmd}")


if __name__ == "__main__":
    sys.exit(main())
