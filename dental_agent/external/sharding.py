"""Shards: the unit of work of a hosted (non-live) Kaggle run.

A hosted task is one pushed source file and one run per model, so the work is cut into ``(condition, image slice)`` tasks. A shard
is small enough that a failure, a spend cap or a slow model costs one slice, never the campaign, and independent enough that
every (model, shard) can run at the same time. ``ShardSpec`` is everything a hosted run needs, embedded in the task source: the
hosted container has no secrets, no annotations and no access to this machine.

Images are named by their path relative to an attached Kaggle dataset, so the same spec works wherever the dataset is mounted
(``/kaggle/input/<name>`` in hosted runs, ``/kaggle/input/datasets/<owner>/<name>`` in interactive notebooks).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REPO_URL = "https://github.com/rezaxr14/VLM-DENTAL.git"
DEFAULT_SHARD_SIZE = 25
DEFAULT_WORKERS = 4

# What a worker must be able to IMPORT. The scoring helpers live in ``dental_agent.evaluation`` whose package ``__init__`` pulls in the
# whole project (torch, torchvision, langgraph, scikit-learn, tqdm, ...), so even a zero-shot shard needs them; a test walks the real
# import chain and fails when this list stops covering it. pip names, not import names.
BASE_REQUIREMENTS = ["pyyaml", "pandas", "pillow", "numpy", "matplotlib", "python-dotenv", "tabulate", "scikit-learn", "tqdm",
                     "huggingface_hub", "langgraph"]
# Hosted containers have no torch. The CPU wheel is tried first (the default PyPI wheel drags in CUDA), the default wheel is the
# fallback, and both are skipped when torch + torchvision are already importable. ``locate_tooth`` additionally needs ultralytics.
TORCH_INSTALL_STEPS = [
    {"args": ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"], "optional": True, "skip_if_importable": "torchvision"},
    {"args": ["torch", "torchvision"], "optional": False, "skip_if_importable": "torchvision"},
]
GROUNDER_INSTALL_STEPS = [{"args": ["ultralytics"], "optional": False, "skip_if_importable": "ultralytics"}]
TOOLS_INSTALL_STEPS = TORCH_INSTALL_STEPS + GROUNDER_INSTALL_STEPS      # kept for callers that name the full tools set


def short_condition(condition: str) -> str:
    return condition.removeprefix("kaggle_").replace("_", "-")


def plan_shards(image_ids, shard_size: int = DEFAULT_SHARD_SIZE) -> list[list[int]]:
    """Contiguous slices of the sorted ids (stable across runs: shard k always holds the same images)."""
    ids = sorted(int(i) for i in image_ids)
    size = max(1, int(shard_size))
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def task_name(cohort_sha: str, condition: str, index: int, remainder: tuple[str, int] | None = None) -> str:
    """``vlmdental-<cohort6>-<condition>-s<NN>``; a remainder task (the images one model still lacks) is ``...-r<model6>-<k>``."""
    base = f"vlmdental-{cohort_sha[:6]}-{short_condition(condition)}-s{index:02d}"
    if remainder is not None:
        base += f"-r{remainder[0][:6]}{remainder[1]}"
    return base.lower()


@dataclass
class ShardSpec:
    task_name: str
    condition: str
    image_ids: list[int]
    rel_paths: dict[str, str]                       # image id -> path relative to the dataset that holds it
    image_datasets: list[str]                       # Kaggle dataset ids that hold the images ("owner/name")
    cohort_sha256: str                              # of the WHOLE cohort (one identity for every shard of a cell)
    split: str
    dataset: str = "dentex"
    extra_datasets: list[str] = field(default_factory=list)      # e.g. the grounder weights
    workers: int = DEFAULT_WORKERS
    overrides: dict[str, Any] = field(default_factory=dict)
    repo_url: str = DEFAULT_REPO_URL
    repo_ref: str | None = None                     # None: the repository's current default branch (not pinned)
    repo_path: str | None = None                    # pre-staged code AND environment (tests): use this folder, clone and install nothing
    requirements: list[str] = field(default_factory=lambda: list(BASE_REQUIREMENTS))
    install_steps: list[dict[str, Any]] = field(default_factory=list)
    needs_grounder: bool = False
    pack: str = "default"                           # the condition's tool pack (recorded for the report; the condition decides)
    plugins: list[str] = field(default_factory=list)   # modules imported before the run (they register packs / conditions / evaluators)
    env_dataset: str | None = None                  # PRIVATE Kaggle dataset holding only a .env (tokens never enter the task source)
    hf_results_repo: str | None = None              # existing Hub repo the worker pushes its finished shard to (needs HF_TOKEN in the .env)
    hf_repo_type: str | None = None                 # "model" | "dataset" (None: resolved from the repo)
    cohort_settings: dict[str, Any] | None = None   # the 250-image cohort's COCO/mapping settings (``cohort.load_split``); part of every run's identity

    @property
    def attach(self) -> list[str]:
        return list(dict.fromkeys([*self.image_datasets, *self.extra_datasets, *([self.env_dataset] if self.env_dataset else [])]))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ShardSpec":
        return cls(**json.loads(text))

    def sha(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


def relative_paths(images_df, image_ids, dataset_root: str | Path) -> dict[str, str]:
    """Path of every image relative to ``dataset_root`` (the folder the Kaggle dataset was created from)."""
    root = Path(dataset_root).resolve()
    out: dict[str, str] = {}
    for row in images_df[images_df["id"].isin([int(i) for i in image_ids])].itertuples():
        out[str(int(row.id))] = Path(row.local_path).resolve().relative_to(root).as_posix()
    missing = sorted(set(str(int(i)) for i in image_ids) - set(out))
    if missing:
        raise ValueError(f"images not found in the image table: {missing[:10]}")
    return out


def build_specs(*, condition: str, image_ids, rel_paths: dict[str, str], image_datasets: list[str], cohort_sha256: str, split: str,
                shard_size: int = DEFAULT_SHARD_SIZE, dataset: str = "dentex", extra_datasets: list[str] | None = None,
                workers: int = DEFAULT_WORKERS, overrides: dict[str, Any] | None = None, repo_url: str = DEFAULT_REPO_URL,
                repo_ref: str | None = None, repo_path: str | None = None,
                needs_grounder: bool | None = None, env_dataset: str | None = None, hf_results_repo: str | None = None,
                hf_repo_type: str | None = None, plugins: list[str] | None = None,
                cohort_settings: dict[str, Any] | None = None) -> list[ShardSpec]:
    from dental_agent.external.conditions import get_condition
    cond = get_condition(condition)
    grounder = cond.with_tools if needs_grounder is None else bool(needs_grounder)
    requirements = list(BASE_REQUIREMENTS)
    specs = []
    for k, ids in enumerate(plan_shards(image_ids, shard_size)):
        specs.append(ShardSpec(
            task_name=task_name(cohort_sha256, condition, k), condition=condition, image_ids=ids,
            rel_paths={str(i): rel_paths[str(i)] for i in ids}, image_datasets=list(image_datasets),
            cohort_sha256=cohort_sha256, split=split, dataset=dataset, extra_datasets=list(extra_datasets or []),
            workers=workers, overrides=dict(overrides or {}), repo_url=repo_url, repo_ref=repo_ref, repo_path=repo_path,
            requirements=list(requirements), install_steps=[dict(st) for st in (TORCH_INSTALL_STEPS + (GROUNDER_INSTALL_STEPS if grounder else []))],
            needs_grounder=grounder, pack=cond.pack, plugins=list(plugins or []), env_dataset=env_dataset, hf_results_repo=hf_results_repo, hf_repo_type=hf_repo_type,
            cohort_settings=dict(cohort_settings) if cohort_settings else None))
    return specs


def specs_for_conditions(conditions: list[str], *, image_ids, rel_paths: dict[str, str], **kw) -> list[ShardSpec]:
    """Shard specs for several conditions over the same cohort (one set of shards per condition)."""
    return [spec for c in conditions for spec in build_specs(condition=c, image_ids=image_ids, rel_paths=rel_paths, **kw)]


def locate_image_datasets(images_df, image_ids, dataset_dirs) -> tuple[list[str], dict[str, str]]:
    """Which attached Kaggle datasets hold the cohort's images, as ``owner/name`` ids read from where they are mounted
    (``<input>/datasets/<owner>/<name>``), and each image's path relative to its dataset. Nothing has to be typed or guessed:
    the notebook already resolved the images, this only reads back where they live."""
    dirs = [Path(d).resolve() for d in dataset_dirs]
    ids_used: dict[Path, None] = {}
    rel: dict[str, str] = {}
    for row in images_df[images_df["id"].isin([int(i) for i in image_ids])].itertuples():
        path = Path(row.local_path).resolve()
        home = next((d for d in dirs if d == path or d in path.parents), None)
        if home is None:
            raise ValueError(f"image {int(row.id)} ({path}) is not inside an attached Kaggle dataset, so a hosted run cannot see it")
        ids_used[home] = None
        rel[str(int(row.id))] = path.relative_to(home).as_posix()
    missing = sorted(set(str(int(i)) for i in image_ids) - set(rel))
    if missing:
        raise ValueError(f"images not found in the image table: {missing[:10]}")
    names = []
    for home in ids_used:
        if home.parent.parent.name == "datasets":
            names.append(f"{home.parent.name}/{home.name}")
        else:
            raise ValueError(f"cannot tell the owner of the dataset mounted at {home}; pass its 'owner/name' id explicitly")
    return names, rel


TASK_TEMPLATE = r'''# Generated by dental_agent.external.sharding; do not edit. One hosted task = one shard of one condition.
import importlib, importlib.util, json, os, subprocess, sys, time, traceback
import kaggle_benchmarks as kbench

SPEC = json.loads(__SPEC_JSON__)
WORKING = os.environ.get("VLM_WORKING_DIR", "/kaggle/working")


def sh(args, timeout=1800):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return {"cmd": " ".join(args)[:200], "rc": r.returncode, "s": round(time.time() - t, 1), "tail": (r.stdout + r.stderr).strip()[-400:]}
    except Exception as exc:
        return {"cmd": " ".join(args)[:200], "rc": -1, "s": round(time.time() - t, 1), "tail": type(exc).__name__ + ": " + str(exc)[:300]}


def bootstrap():
    steps = []
    repo = SPEC.get("repo_path")
    if not repo or not os.path.isdir(os.path.join(repo, "dental_agent")):
        repo = "/tmp/vlm-dental"
        if not os.path.isdir(os.path.join(repo, "dental_agent")):
            ref = ["--branch", SPEC["repo_ref"]] if SPEC.get("repo_ref") else []
            steps.append(sh(["git", "clone", "--depth", "1", *ref, SPEC["repo_url"], repo], 600))
    sha = sh(["git", "-C", repo, "rev-parse", "HEAD"], 30)["tail"].splitlines()[-1:] or [""]
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    prestaged = bool(SPEC.get("repo_path")) and repo == SPEC["repo_path"]       # a pre-staged checkout means a pre-staged environment
    if SPEC.get("requirements") and not prestaged:
        steps.append(sh(pip + SPEC["requirements"], 900))
    for step in ([] if prestaged else SPEC.get("install_steps", [])):
        importlib.invalidate_caches()
        if step.get("skip_if_importable") and importlib.util.find_spec(step["skip_if_importable"]) is not None:
            continue
        res = sh(pip + step["args"], 1800)
        steps.append(res)
        if res["rc"] != 0 and not step.get("optional"):
            break
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return {"repo": repo, "repo_sha": sha[0].strip(), "steps": steps, "python": sys.version.split()[0]}


@kbench.task(name=SPEC["task_name"], description="VLM-DENTAL external benchmark shard: " + SPEC["condition"])
def shard(llm) -> None:
    info = bootstrap()
    try:
        from dental_agent.external.hosted_runtime import run_shard
    except Exception as exc:                  # the worker cannot import the project: say why in a manifest the driver can read
        os.makedirs(WORKING, exist_ok=True)
        with open(os.path.join(WORKING, "shard_manifest.json"), "w") as f:
            json.dump({"task_name": SPEC["task_name"], "condition": SPEC["condition"], "status": "failed", "stage": "import",
                       "error": (type(exc).__name__ + ": " + str(exc))[:800], "traceback": traceback.format_exc()[-1500:],
                       "bootstrap": info}, f, indent=2)
        raise
    run_shard(SPEC, str(getattr(llm, "model", None) or getattr(llm, "name", "")), bootstrap_info=info)


shard.run(kbench.llm)
'''


def render_task_source(spec: ShardSpec) -> str:
    """The Python file pushed to Kaggle for this shard. Compiles (checked) and carries the whole spec."""
    source = TASK_TEMPLATE.replace("__SPEC_JSON__", repr(spec.to_json()))
    compile(source, f"{spec.task_name}.py", "exec")
    return source
