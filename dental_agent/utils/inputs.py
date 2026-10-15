"""One resolver per input a SFT / GRPO run needs, each with a FIXED search order.

    explicit flag  ->  local candidates  ->  Kaggle inputs  ->  Hugging Face Hub

Every resolver either returns where the input is (and how it was found) or raises :class:`InputNotFound` naming exactly
what was searched and the single command that fixes it. The run scripts call them BEFORE any model is loaded, so a
missing adapter / trace / image / checkpoint costs seconds, not a 9B model load.

Rules shared by all resolvers
* Discovery resolves PATHS only. It never changes a setting: a precision, a canonical-resize flag or a max-seq-len is
  never inferred from what is found on disk.
* Relative paths are tried against the working directory AND the repository root, so a notebook that ``%cd``-ed
  somewhere else, or a script started from another folder, finds the same files.
* The ``hub`` argument is ``"download"`` (fetch what is missing), ``"probe"`` (only look whether the Hub has it, used by
  the read-only ``--preflight``) or ``"off"`` (local only).
* A trace image of the DENTEX *train* pool is never resolved to a validation (= test) file: image ids collide between
  the two sets (1-705 vs 1-50), so a missing train image must fail, not silently become a test image.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")

DEFAULT_MODELS_REPO = "Reza-Nadimi/vlm-dental-models"
DEFAULT_TRACES_REPO = "Reza-Nadimi/vlm-dental-traces"
DEFAULT_IMAGES_REPOS = {
    "dentex": "Reza-Nadimi/dentex-train-images",
    "tufts": "Reza-Nadimi/tufts-train-images",
}
YOLO_HUB_FILE = "yolo_cv/yolo_cv_best/weights/best.pt"
YOLO_DEFAULT_PATH = Path("data/models/yolo_cv_best/weights/best.pt")
MANIFEST_NAME = "trace_token_lengths.json"
FINGERPRINT_VERSION = 1

HubMode = Literal["download", "probe", "off"]


class InputNotFound(RuntimeError):
    """An input could not be found. The message lists every place that was searched and the one command that fixes it."""

    def __init__(self, what: str, searched: Sequence[str], fix: str) -> None:
        self.what = what
        self.searched = list(searched)
        self.fix = fix
        lines = [f"{what} not found."]
        if self.searched:
            lines.append("  searched:")
            lines += [f"    - {s}" for s in self.searched]
        lines.append(f"  fix: {fix}")
        super().__init__("\n".join(lines))


@dataclass
class Resolved:
    """Where an input was found. ``source``: flag | local | kaggle | hub-cache | hub (fetched) | hub-available (probe)."""

    path: Path
    source: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.path} [{self.source}]"


# --------------------------------------------------------------------------------------------------------------------
# Hub access (three thin functions so tests can replace them; nothing else in this module touches the network)
# --------------------------------------------------------------------------------------------------------------------
def hub_token() -> str | None:
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return None if (not t or t.startswith("your_")) else t


def hub_list_files(repo_id: str, repo_type: str = "model") -> list[str]:
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id=repo_id, repo_type=repo_type, token=hub_token()))


def hub_snapshot(repo_id: str, repo_type: str, allow_patterns: Sequence[str], local_dir: str | Path) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id, repo_type=repo_type, allow_patterns=list(allow_patterns), local_dir=str(local_dir), token=hub_token()
    )


def hub_file(repo_id: str, repo_type: str, filename: str, local_dir: str | Path) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=str(local_dir), token=hub_token())


def models_repo(arg: str | None = None) -> str:
    return arg or os.environ.get("HF_ARTIFACT_REPO") or DEFAULT_MODELS_REPO


def traces_repo(arg: str | None = None) -> str:
    return arg or os.environ.get("HF_TRACES_REPO") or DEFAULT_TRACES_REPO


def images_repo(dataset: str, arg: str | None = None) -> str | None:
    ds = (dataset or "dentex").lower().strip()
    env = {"dentex": "DENTEX_IMAGES_REPO", "tufts": "TUFTS_IMAGES_REPO"}.get(ds)
    return arg or (os.environ.get(env) if env else None) or DEFAULT_IMAGES_REPOS.get(ds)


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:140] if str(e) else ''}"


# --------------------------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------------------------
def as_candidates(p: str | Path) -> list[Path]:
    """A path as given (relative to the working directory) and, when relative, against the repository root."""
    p = Path(p)
    return [p] if p.is_absolute() else [p, REPO_ROOT / p]


def search_roots(data_dir: str | Path | None = None) -> list[Path]:
    """Existing-or-not data roots, most specific first, independent of the working directory."""
    roots: list[Path] = []

    def add(p: Path) -> None:
        for q in as_candidates(p):
            if q not in roots:
                roots.append(q)

    if data_dir:
        add(Path(data_dir))
    add(Path("data"))
    for extra in (Path("/kaggle/working/data"), Path("/kaggle/working/VLM-DENTAL/data"), Path("/content/VLM-DENTAL/data")):
        if extra not in roots:
            roots.append(extra)
    return roots


def kaggle_dataset_dirs() -> list[Path]:
    """Attached Kaggle inputs: ``/kaggle/input/<name>`` and ``/kaggle/input/datasets/<owner>/<name>``."""
    out: list[Path] = []
    if not KAGGLE_INPUT.is_dir():
        return out
    try:
        for d in sorted(KAGGLE_INPUT.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "datasets":
                for owner in sorted(d.iterdir()):
                    if owner.is_dir():
                        out.extend(sorted(x for x in owner.iterdir() if x.is_dir()))
            else:
                out.append(d)
    except OSError:
        pass
    return out


def _hf_cache_roots() -> list[Path]:
    roots = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


# --------------------------------------------------------------------------------------------------------------------
# Trace images
# --------------------------------------------------------------------------------------------------------------------
_TEST_NAME = re.compile(r"(^|[^a-z])(val|valid|validation|test)([^a-z]|$)", re.IGNORECASE)
_IMG_CACHE: dict[tuple, Path] = {}


def _looks_like_test_file(p: Path) -> bool:
    """True when the file, its folder or its grandfolder is named like the validation / test split."""
    return any(_TEST_NAME.search(part) for part in (p.name, p.parent.name, p.parent.parent.name))


def _norm_id(image_id: Any) -> str:
    try:
        return str(int(image_id))
    except (TypeError, ValueError):
        return str(image_id)


def infer_dataset(rec: dict[str, Any]) -> str:
    ds = str(rec.get("dataset") or "").lower().strip()
    if ds:
        return ds
    origin = str(rec.get("_origin_file", "")).lower()
    return "tufts" if "tufts" in origin else "dentex"


def image_name_candidates(dataset: str, image_id: Any, raw_path: str | None = None) -> list[str]:
    iid = _norm_id(image_id)
    fname = Path(str(raw_path).replace("\\", "/")).name if raw_path else ""
    if dataset == "tufts":
        names = [fname, f"{iid}.JPG", f"{iid}.jpg", f"{iid}.jpeg", f"{iid}.png"]
    else:
        names = [fname, f"{iid}.png", f"train_{iid}.png", f"{iid}.jpg"]
    return list(dict.fromkeys(n for n in names if n))


def _image_dirs(dataset: str, root: Path) -> list[Path]:
    if dataset == "tufts":
        rels = ["tufts/Radiographs", "Tufts/Radiographs", "tufts/images", "Tufts/images", "Tufts", "tufts", "Radiographs"]
        globs: list[str] = []
    else:
        rels = ["dentex/images", "dentex", "images", "dentex/train_images", "train_images"]
        globs = ["dentex/DENTEX/training_data/*/xrays", "dentex/training_data/*/xrays", "training_data/*/xrays"]
    dirs = [root / r for r in rels]
    for pat in globs:
        try:
            dirs.extend(sorted(root.glob(pat)))
        except OSError:
            pass
    return dirs


def resolve_image(
    dataset: str,
    image_id: Any,
    raw_path: str | None = None,
    data_dir: str | Path | None = None,
    *,
    split: str = "train",
    allow_hub: bool = True,
    repo: str | None = None,
) -> Path | None:
    """Find the file of a trace image whatever the working directory or the machine that recorded ``raw_path``.

    Order: the path as recorded -> that path rebased on the data roots (Windows separators, ``/kaggle/working/...`` and
    other machines' prefixes dropped) -> ``<dir>/<id>.png`` style names in the data roots and attached Kaggle inputs ->
    a recursive search of the dataset folder -> the local Hugging Face cache -> a one-image Hub download.
    """
    ds = (dataset or "dentex").lower().strip()
    raw = str(raw_path).strip() if raw_path else ""
    raw_norm = raw.replace("\\", "/")
    key = (ds, _norm_id(image_id), raw, str(data_dir), split)
    hit = _IMG_CACHE.get(key)
    if hit is not None and hit.is_file():
        return hit

    guard = ds == "dentex" and split in ("train", "training")

    def ok(p: Path) -> bool:
        return p.is_file() and not (guard and _looks_like_test_file(p))

    def found(p: Path) -> Path:
        _IMG_CACHE[key] = p
        return p

    roots = search_roots(data_dir)
    names = image_name_candidates(ds, image_id, raw)

    if raw:
        for c in [Path(raw)] + ([Path(raw_norm)] if raw_norm != raw else []):
            if ok(c):
                return found(c)
        if not raw_norm.startswith("/") and not re.match(r"^[A-Za-z]:/", raw_norm):
            for base in [Path.cwd(), REPO_ROOT] + roots + [r.parent for r in roots]:
                c = base / raw_norm
                if ok(c):
                    return found(c)
        parts = [p for p in raw_norm.split("/") if p]
        for anchor in ("images", "Radiographs", "xrays", "training_data"):
            if anchor in parts:
                tail = Path(*parts[parts.index(anchor):])
                for root in roots:
                    for sub in ("", ds, ds.capitalize()):
                        c = root / sub / tail
                        if ok(c):
                            return found(c)

    kaggle = kaggle_dataset_dirs()
    for base in roots:
        for d in _image_dirs(ds, base):
            if d.is_dir():
                for n in names:
                    if ok(d / n):
                        return found(d / n)
    for kd in kaggle:
        for d in [kd, kd / "images", kd / "Radiographs", kd / ds / "images", kd / ds / "Radiographs"]:
            if d.is_dir():
                for n in names:
                    if ok(d / n):
                        return found(d / n)

    for base in roots + kaggle:
        for sub in (base / ds, base / ds.capitalize(), base):
            if not sub.is_dir() or (sub == base and base in roots):
                continue
            for n in names:
                try:
                    for m in sub.rglob(n):
                        if ok(m):
                            return found(m)
                except OSError:
                    continue

    for cache_root in roots + _hf_cache_roots():
        try:
            for hub_dir in cache_root.glob("datasets--*"):
                if ds not in hub_dir.name.lower():
                    continue
                for n in names:
                    for sub in ("images", "Radiographs"):
                        for m in hub_dir.glob(f"snapshots/*/{sub}/{n}"):
                            if ok(m):
                                return found(m)
        except OSError:
            continue

    if allow_hub and _norm_id(image_id).isdigit():
        got = _download_slice(ds, [int(_norm_id(image_id))], data_dir, repo)
        p = got.get(int(_norm_id(image_id)))
        if p is not None and ok(Path(p)):
            return found(Path(p))
    return None


def _download_slice(ds: str, ids: list[int], data_dir: str | Path | None, repo: str | None) -> dict[int, Path | None]:
    rid = images_repo(ds, repo)
    if not rid or not ids:
        return {}
    cache = str(data_dir) if data_dir else None
    try:
        if ds == "tufts":
            from dental_agent.data.tufts import download_tufts_slice

            return download_tufts_slice(ids, repo_id=rid, cache_dir=cache)
        from dental_agent.data.dentex import download_dentex_slice

        return download_dentex_slice(ids, repo_id=rid, cache_dir=cache, split_name="train")
    except Exception as e:  # network / auth problems surface as "still missing" with the reason in the message
        print(f"[IMAGES] Hub fetch from {rid} failed: {_err(e)}")
        return {}


@dataclass
class ImageCheck:
    """Result of resolving every image a set of traces points to."""

    n_records: int = 0
    n_unique: int = 0
    fetched: int = 0
    missing: list[tuple[str, str, str]] = field(default_factory=list)  # (dataset, image_id, path as recorded)

    @property
    def n_missing(self) -> int:
        return len(self.missing)

    def by_dataset(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ds, _, _ in self.missing:
            out[ds] = out.get(ds, 0) + 1
        return out

    def describe(self, k: int = 3) -> str:
        ex = ", ".join(f"{ds}::{iid}" for ds, iid, _ in self.missing[:k])
        return f"{self.n_missing} of {self.n_unique} images missing ({self.by_dataset()}), e.g. {ex}"


def iter_trace_records(files: Iterable[str | Path], exclude_keys: set[str] | None = None) -> Iterator[dict[str, Any]]:
    """Light view of every trace: dataset, image_id, image_path, _origin_file. Dev-split keys are skipped."""
    for f in files:
        p = Path(f)
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                rec = {"dataset": r.get("dataset"), "image_id": r.get("image_id"), "image_path": r.get("image_path"), "_origin_file": p.name}
                rec["dataset"] = infer_dataset(rec)
                if exclude_keys and f"{rec['dataset']}::{_norm_id(rec['image_id'])}" in exclude_keys:
                    continue
                yield rec


def ensure_images(
    items: Iterable[tuple[str, Any, str | None]],
    data_dir: str | Path | None = None,
    *,
    hub: HubMode = "download",
    split: str = "train",
    repo: str | None = None,
) -> ImageCheck:
    """Resolve every ``(dataset, image_id, recorded_path)``; fetch what is missing from the Hub in ONE call per dataset."""
    check = ImageCheck()
    seen: dict[tuple[str, str], str | None] = {}
    for ds, iid, raw in items:
        check.n_records += 1
        seen.setdefault((str(ds).lower(), _norm_id(iid)), raw)
    check.n_unique = len(seen)
    missing = [(ds, iid, raw) for (ds, iid), raw in seen.items() if resolve_image(ds, iid, raw, data_dir, split=split, allow_hub=False) is None]
    if missing and hub == "download":
        by_ds: dict[str, list[int]] = {}
        for ds, iid, _ in missing:
            if iid.isdigit():
                by_ds.setdefault(ds, []).append(int(iid))
        for ds, ids in by_ds.items():
            _download_slice(ds, ids, data_dir, repo)
        still = [(ds, iid, raw) for ds, iid, raw in missing if resolve_image(ds, iid, raw, data_dir, split=split, allow_hub=False) is None]
        check.fetched = len(missing) - len(still)
        missing = still
    check.missing = [(ds, iid, raw or "") for ds, iid, raw in missing]
    return check


def check_trace_images(
    files: Iterable[str | Path], data_dir: str | Path | None = None, *, exclude_keys: set[str] | None = None, hub: HubMode = "off"
) -> ImageCheck:
    """Count the trace images that do not resolve (``hub="off"`` is the read-only check used by ``--preflight``)."""
    items = [(r["dataset"], r["image_id"], r["image_path"]) for r in iter_trace_records(files, exclude_keys)]
    return ensure_images(items, data_dir, hub=hub)


def ensure_trace_images(
    files: Iterable[str | Path], data_dir: str | Path | None = None, *, exclude_keys: set[str] | None = None
) -> ImageCheck:
    """Resolve (and fetch from the Hub) every trace image before any model is loaded; raise if any stays missing."""
    check = check_trace_images(files, data_dir, exclude_keys=exclude_keys, hub="download")
    if check.n_missing:
        raise InputNotFound(
            f"Trace images ({check.describe()})",
            [f"data roots {[str(r) for r in search_roots(data_dir)[:3]]}", "attached Kaggle inputs", "local Hugging Face cache",
             *[f"Hub dataset {images_repo(ds)}" for ds in check.by_dataset()]],
            "python scripts/fetch_hf_assets.py --images   (or set DENTEX_IMAGES_REPO / TUFTS_IMAGES_REPO and HF_TOKEN, "
            "or attach the Kaggle image dataset)",
        )
    return check


def ensure_dataset_images(images_df: Any, dataset: str, data_dir: str | Path | None, *, split: str = "train", hub: HubMode = "download"):
    """Fill/repair ``local_path`` of a loader dataframe, fetching missing images from the Hub. Returns ``(df, ImageCheck)``.

    ``load_dentex_dataset`` can return rows without a file (annotation JSON came from the Hub, images did not); the GRPO
    pool then silently dropped them. This makes that visible and repairs it.
    """
    df = images_df.copy()
    if "local_path" not in df.columns:
        df["local_path"] = None
    ds = (dataset or "dentex").lower().strip()
    rows = [(i, str(r["id"]), r.get("file_name"), r.get("local_path")) for i, r in df.iterrows()]
    need = [(i, iid, fn) for i, iid, fn, lp in rows if not (isinstance(lp, str) and Path(lp).is_file())]
    check = ensure_images([(ds, iid, fn) for _, iid, fn in need], data_dir, hub=hub, split=split)
    for i, iid, fn in need:
        p = resolve_image(ds, iid, fn, data_dir, split=split, allow_hub=False)
        if p is not None:
            df.at[i, "local_path"] = str(p)
    check.n_records = len(rows)
    return df, check


def hub_probe(repo_id: str, repo_type: str, paths: Sequence[str]) -> tuple[list[str], str | None]:
    """Read-only: which of ``paths`` (files or folder prefixes) exist in a Hub repo. Returns ``(present, error)``; ``error`` is
    set when the repo could not be listed (offline, private without a token), in which case nothing can be said."""
    try:
        files = hub_list_files(repo_id, repo_type)
    except Exception as e:
        return [], _err(e)
    return [p for p in paths if any(f == p or f.startswith(p.rstrip("/") + "/") for f in files)], None


def hub_image_filename(dataset: str, image_id: Any) -> str:
    """File of an image inside its per-image Hub dataset repo (see scripts/upload_dataset_images_to_hf.py)."""
    return f"Radiographs/{_norm_id(image_id)}.JPG" if dataset == "tufts" else f"images/{_norm_id(image_id)}.png"


def probe_images_on_hub(missing: Sequence[tuple[str, str, str]]) -> tuple[int, int, str | None]:
    """Of locally missing images ``(dataset, id, path)``: how many the Hub repos hold, how many nobody holds, and an error."""
    fetchable = unavailable = 0
    for ds in sorted({m[0] for m in missing}):
        rid = images_repo(ds)
        ids = [m for m in missing if m[0] == ds]
        if not rid:
            unavailable += len(ids)
            continue
        try:
            files = set(hub_list_files(rid, "dataset"))
        except Exception as e:
            return fetchable, unavailable, f"{rid}: {_err(e)}"
        for _, iid, _ in ids:
            if hub_image_filename(ds, iid) in files:
                fetchable += 1
            else:
                unavailable += 1
    return fetchable, unavailable, None


# --------------------------------------------------------------------------------------------------------------------
# SFT adapter (input of GRPO and of the K sweep)
# --------------------------------------------------------------------------------------------------------------------
def sft_adapter_name(track: str, stage: str, precision: str | None = None) -> str:
    base = f"qwen3_5_9b_sft_{track}_{stage}"
    return f"{base}_{precision}" if precision else base


def sft_adapter_candidates(track: str, stage: str, precision: str | None = None) -> list[Path]:
    """Local folders searched for the SFT adapter. ``train_sft.py`` writes ``..._<precision>``; with ``precision`` (the
    run's own explicit ``--precision``, never guessed) that folder is first. The other precision's folder is never a
    candidate: a bf16 run does not pick up a QLoRA adapter and nothing switches it."""
    cands = [Path(f"data/models/{sft_adapter_name(track, stage, precision)}")] if precision else []
    cands += [
        Path(f"data/models/qwen3_5_9b_sft_{track}_{stage}"),
        Path(f"checkpoints/sft_{track}_{stage}"),
        Path(f"data/models/qwen3_5_9b_sft_{track}"),
    ]
    if precision:  # where a Hub download of this resolver lands
        cands.append(Path(f"data/models/sft/{sft_adapter_name(track, stage, precision)}"))
    cands.append(Path(f"data/models/sft/qwen3_5_9b_sft_{track}_{stage}"))
    return cands


def has_adapter_config(p: Path) -> bool:
    return (p / "adapter_config.json").is_file()


def adapter_problem(p: Path) -> str | None:
    """Why a folder cannot be loaded as a LoRA adapter, or None."""
    if not p.is_dir():
        return "is not a folder"
    if not has_adapter_config(p):
        hint = " (it has a best_adapter/ subfolder: pass that folder or copy it up)" if (p / "best_adapter").is_dir() else ""
        return f"has no adapter_config.json{hint}"
    if not ((p / "adapter_model.safetensors").is_file() or (p / "adapter_model.bin").is_file()):
        return "has adapter_config.json but no adapter_model.safetensors / adapter_model.bin"
    return None


def assert_adapter_loadable(p: str | Path) -> None:
    prob = adapter_problem(Path(p))
    if prob:
        raise InputNotFound(f"Loadable SFT adapter at {p}", [f"{p} {prob}"],
                            "pass --sft-model-dir <folder that holds adapter_config.json and adapter_model.safetensors>")


def _sft_meta_matches(d: Path, track: str, stage: str, precision: str | None) -> bool:
    from dental_agent.training.run_meta import read_sft_run_meta

    meta = read_sft_run_meta(d)
    if not meta:
        return False
    return meta.get("track") == track and meta.get("stage") == stage and (precision is None or meta.get("precision") == precision)


def resolve_sft_adapter(
    explicit: str | None, track: str, stage: str, precision: str | None = None, hf_repo: str | None = None, *, hub: HubMode = "download"
) -> Resolved:
    """flag -> local folders (incl. the ``_<precision>`` one) -> Kaggle inputs -> Hub ``sft/<name>``.

    An explicit ``--sft-model-dir`` that is wrong FAILS; it never falls through to some other adapter.
    """
    fix_train = (f"train it first (python scripts/train_sft.py --track {track} --stage {stage}"
                 f"{f' --precision {precision}' if precision else ''} ...), or pass --sft-model-dir <folder with adapter_config.json>")
    if explicit:
        cands = as_candidates(explicit)
        for c in cands:
            if has_adapter_config(c):
                return Resolved(c, "flag")
        why = "exists but has no adapter_config.json" if any(c.is_dir() for c in cands) else "no such folder"
        raise InputNotFound("SFT adapter", [f"--sft-model-dir {explicit}: {why}"],
                            "pass the folder that holds adapter_config.json (train_sft.py writes "
                            f"data/models/{sft_adapter_name(track, stage, precision or '<precision>')})")

    searched: list[str] = []
    for cand in sft_adapter_candidates(track, stage, precision):
        for c in as_candidates(cand):
            if has_adapter_config(c):
                return Resolved(c, "local")
        searched.append(str(cand))

    names = list(dict.fromkeys([sft_adapter_name(track, stage, precision), sft_adapter_name(track, stage)]))
    if KAGGLE_INPUT.is_dir():
        for kd in kaggle_dataset_dirs():
            for nm in names:
                for c in (kd / nm, kd / "sft" / nm):
                    if has_adapter_config(c):
                        return Resolved(c, "kaggle")
            if has_adapter_config(kd) and _sft_meta_matches(kd, track, stage, precision):
                return Resolved(kd, "kaggle")
        searched.append(f"{KAGGLE_INPUT}/<dataset>/{names[0]} (and /kaggle/input/datasets/<owner>/<dataset>/...)")

    repo = models_repo(hf_repo)
    if hub != "off" and repo:
        try:
            files = set(hub_list_files(repo, "model"))
        except Exception as e:
            searched.append(f"Hub {repo}: unreachable ({_err(e)})")
        else:
            for nm in names:
                prefix = f"sft/{nm}/"
                if f"{prefix}adapter_config.json" not in files:
                    continue
                if hub == "probe":
                    return Resolved(Path(f"hub://{repo}/sft/{nm}"), "hub-available")
                hub_snapshot(repo, "model", [prefix + "adapter_config.json", prefix + "adapter_model.*", prefix + "sft_run_meta.json"], "data/models")
                dest = Path("data/models") / "sft" / nm
                if has_adapter_config(dest):
                    return Resolved(dest, "hub")
                searched.append(f"Hub {repo}/sft/{nm}: downloaded but no adapter_config.json arrived")
            searched.append(f"Hub {repo}: sft/{names[0]}/ " + (f"and sft/{names[-1]}/ " if names[-1] != names[0] else "") + "not present")
    raise InputNotFound("SFT adapter", searched, fix_train)


# --------------------------------------------------------------------------------------------------------------------
# SFT traces
# --------------------------------------------------------------------------------------------------------------------
def stage_trace_names(stage: str, track: str) -> list[str]:
    """Trace files a curriculum stage trains on (disease + negative controls). The single table; train_sft.py uses it."""
    t = track == "with_tools"

    def f(base: str) -> str:
        return f"{base}.jsonl" if t else f"{base}_no_tools.jsonl"

    if stage == "dentex_alone":
        return [f("train_cot_traces_dentex"), f("train_cot_traces_healthy_dentex")]
    if stage == "dentex_tufts_overlap":
        return [f("train_cot_traces_dentex"), f("train_cot_traces_tufts"), f("train_cot_traces_healthy_dentex"), f("train_cot_traces_healthy_tufts")]
    return [f("train_cot_traces_dentex"), f("train_cot_traces_tufts_all"), f("train_cot_traces_healthy_dentex"), f("train_cot_traces_healthy_tufts")]


def trace_dirs(data_dir: str | Path | None) -> list[Path]:
    dirs = [Path(data_dir) / "traces"] if data_dir else []
    dirs += [Path("data/traces"), REPO_ROOT / "data" / "traces"]
    out: list[Path] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


@dataclass
class TraceResolution:
    files: list[Path]
    hub_pending: list[str] = field(default_factory=list)
    legacy_fallback: bool = False


def resolve_stage_traces(stage: str, track: str, data_dir: str | Path | None = "data", repo: str | None = None, *, hub: HubMode = "download") -> TraceResolution:
    """Every trace file of the stage, local first, the missing ones (and only those) from the Hub.

    ALL files of the stage are required: training ``dentex_alone`` on the disease file without the healthy one would
    silently teach the model never to answer "no findings". The old single-file fallback (``train_cot_traces[_no_tools]``)
    is kept for the case where NO stage file exists anywhere.
    """
    names = stage_trace_names(stage, track)
    dirs = trace_dirs(data_dir)
    legacy = "train_cot_traces.jsonl" if track == "with_tools" else "train_cot_traces_no_tools.jsonl"
    found: dict[str, Path] = {}
    for n in names:
        for d in dirs:
            if (d / n).is_file():
                found[n] = d / n
                break
    missing = [n for n in names if n not in found]
    pending: list[str] = []
    searched = [f"{d}/{n}" for n in missing for d in dirs[:2]]
    if missing and hub != "off":
        rid = traces_repo(repo)
        try:
            remote = set(hub_list_files(rid, "dataset"))
        except Exception as e:
            remote = set()
            searched.append(f"Hub {rid}: unreachable ({_err(e)})")
        target = next((d for d in dirs if d.is_dir()), dirs[0])
        for n in list(missing):
            if n not in remote:
                continue
            if hub == "probe":
                pending.append(n)
                missing.remove(n)
                continue
            target.mkdir(parents=True, exist_ok=True)
            hub_file(rid, "dataset", n, target)
            if (target / n).is_file():
                found[n] = target / n
                missing.remove(n)
        if missing:
            searched.append(f"Hub {rid}: lacks {missing}")
    if missing:
        if not found and not pending:
            for d in dirs:
                if (d / legacy).is_file():
                    return TraceResolution([d / legacy], legacy_fallback=True)
        raise InputNotFound(f"Trace files of stage '{stage}' / track '{track}' ({', '.join(missing)})", searched,
                            "python scripts/sync_traces_hf.py --download   (HF_TOKEN set), or generate them with scripts/run_trace_gen.py")
    return TraceResolution([found[n] for n in names if n in found], hub_pending=pending)


def trace_parity(traces_dir: str | Path) -> list[dict[str, Any]]:
    """Per with-tools / no-tools file pair: image keys present in one track and missing in the other."""
    d = Path(traces_dir)
    out = []
    for f in sorted(d.glob("train_cot_traces*.jsonl")):
        if f.name.endswith("_no_tools.jsonl") or "unverified" in f.name:
            continue
        twin = d / f.name.replace(".jsonl", "_no_tools.jsonl")
        if not twin.is_file():
            continue

        def keys(p: Path) -> set[str]:
            return {f"{r['dataset']}::{_norm_id(r['image_id'])}" for r in iter_trace_records([p])}

        a, b = keys(f), keys(twin)
        out.append({"file": f.name, "with_tools": len(a), "no_tools": len(b),
                    "only_with_tools": sorted(a - b), "only_no_tools": sorted(b - a)})
    return out


# --------------------------------------------------------------------------------------------------------------------
# Token-length manifest: found AND checked for staleness
# --------------------------------------------------------------------------------------------------------------------
def _sha1_text_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk.replace(b"\r\n", b"\n"))
    return h.hexdigest()


def code_fingerprint() -> str:
    """Hash of the sources that decide how many tokens a trace has (dataset rendering, canonical sizes, prompts)."""
    h = hashlib.sha1()
    for rel in ("dental_agent/training/sft.py", "dental_agent/utils/canonical.py", "dental_agent/agent/prompts.py"):
        p = REPO_ROOT / rel
        h.update(rel.encode())
        h.update(_sha1_text_file(p).encode() if p.is_file() else b"missing")
    return h.hexdigest()


def manifest_fingerprint(trace_files: Iterable[str | Path], canonical_resize: bool) -> dict[str, Any]:
    traces = {}
    for f in trace_files:
        p = Path(f)
        n = sum(1 for line in open(p, "r", encoding="utf-8") if line.strip())
        traces[p.name] = {"sha1": _sha1_text_file(p), "records": n}
    return {"version": FINGERPRINT_VERSION, "code": code_fingerprint(), "canonical_resize": bool(canonical_resize), "traces": traces}


def manifest_staleness(
    manifest: dict[str, Any], trace_files: Iterable[str | Path], canonical_resize: bool, *, require_complete: bool = True
) -> list[str]:
    """Reasons a manifest no longer describes the current code / traces (empty list = fresh). ``require_complete=False`` is
    for resuming an interrupted measurement, where a partial manifest from the same code and traces is exactly what is wanted."""
    fp = (manifest.get("_meta") or {}).get("fingerprint")
    if not fp:
        return ["it has no fingerprint (written before patch 0034), so it cannot be shown to match the current dataset code and traces"]
    reasons = []
    meta = manifest.get("_meta") or {}
    if require_complete and not meta.get("complete", False):
        reasons.append("the measurement did not finish (an interrupted run leaves a partial manifest)")
    if meta.get("estimated_traces"):
        reasons.append(f"{len(meta['estimated_traces'])} entries are estimates, not measurements")
    if bool(manifest.get("canonical_resize", False)) != bool(canonical_resize):
        reasons.append(f"it was measured with canonical_resize={bool(manifest.get('canonical_resize', False))}, this run uses {bool(canonical_resize)}")
    if fp.get("code") != code_fingerprint():
        reasons.append("the dataset / canonical / prompt code changed since it was measured")
    cur = manifest_fingerprint(trace_files, canonical_resize)["traces"]
    for name, info in cur.items():
        old = (fp.get("traces") or {}).get(name)
        if old is None:
            reasons.append(f"{name} was not part of it")
        elif old.get("sha1") != info["sha1"]:
            reasons.append(f"{name} changed ({old.get('records')} -> {info['records']} traces)")
    return reasons


def manifest_candidates(data_dir: str | Path | None = "data", explicit: str | Path | None = None) -> list[Path]:
    cands = [Path(explicit)] if explicit else []
    cands += [d / MANIFEST_NAME for d in trace_dirs(data_dir)] + [Path("/kaggle/working/data/traces") / MANIFEST_NAME]
    out: list[Path] = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


@dataclass
class ManifestStatus:
    path: Path | None
    state: str  # "fresh" | "stale" | "missing"
    reasons: list[str] = field(default_factory=list)


def check_token_manifest(
    data_dir: str | Path | None, trace_files: Iterable[str | Path], canonical_resize: bool, explicit: str | Path | None = None
) -> ManifestStatus:
    cands = manifest_candidates(data_dir, explicit)
    for c in cands:
        if c.is_file():
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
            except Exception as e:
                return ManifestStatus(c, "stale", [f"it is not readable JSON ({_err(e)})"])
            reasons = manifest_staleness(data, list(trace_files), canonical_resize)
            return ManifestStatus(c, "stale" if reasons else "fresh", reasons)
    return ManifestStatus(None, "missing", [f"not found in {', '.join(str(c) for c in cands[:3])}"])


def manifest_command(data_dir: str | Path, canonical_resize: bool, model_id: str, vision_slots: Sequence[int]) -> str:
    return (f"python scripts/compute_exact_trace_lengths.py --model-id {model_id} --data-dir {data_dir} --recompute "
            f"--vision-slots {' '.join(map(str, vision_slots))}" + (" --canonical-resize" if canonical_resize else ""))


# --------------------------------------------------------------------------------------------------------------------
# Dev split
# --------------------------------------------------------------------------------------------------------------------
def resolve_dev_split(path: str | Path | None, data_dir: str | Path | None = None) -> Path | None:
    """The dev-split JSON (committed in git as data/splits/dentex_dev.json); ``None`` = nothing is held out."""
    if path is None:
        return None
    cands: list[Path] = []
    for c in as_candidates(path):
        cands.append(c)
    for r in search_roots(data_dir)[:4]:
        cands.append(r / "splits" / Path(path).name)
    for c in cands:
        if c.is_file():
            return c
    raise InputNotFound("Dev split", [str(c) for c in cands[:4]],
                        f"python scripts/make_dev_split.py --data-dir {data_dir or 'data'} --n 70 --seed 42 --output {path}   "
                        "(or git pull: data/splits/dentex_dev.json is committed)")


# --------------------------------------------------------------------------------------------------------------------
# YOLO grounding checkpoint
# --------------------------------------------------------------------------------------------------------------------
def yolo_candidates(explicit: str | Path | None = None) -> list[Path]:
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("GROUNDING_MODEL_PATH"):
        cands.append(Path(os.environ["GROUNDING_MODEL_PATH"]))
    cands += [
        YOLO_DEFAULT_PATH,
        Path("data/models/grounding_tool_cv_best/weights/best.pt"),
        Path("data/models/dentex_grounding_tool_cv_best/weights/best.pt"),
        Path("data/models/yolo_cv/yolo_cv_best/weights/best.pt"),
    ]
    return list(dict.fromkeys(cands))


def resolve_yolo_checkpoint(explicit: str | Path | None = None, repo: str | None = None, *, hub: HubMode = "download") -> Resolved:
    """GROUNDING_MODEL_PATH / local folders / Kaggle inputs / Hub ``yolo_cv/yolo_cv_best/weights/best.pt``."""
    searched: list[str] = []
    for cand in yolo_candidates(explicit):
        for c in as_candidates(cand):
            if c.is_file():
                return Resolved(c, "flag" if explicit and Path(explicit) == cand else "local")
        searched.append(str(cand))
    if KAGGLE_INPUT.is_dir():
        for kd in kaggle_dataset_dirs():
            for rel in ("yolo_cv_best/weights/best.pt", "weights/best.pt", "yolo_cv/yolo_cv_best/weights/best.pt", "best.pt"):
                if (kd / rel).is_file() and any(w in kd.name.lower() for w in ("yolo", "ground", "tooth")):
                    return Resolved(kd / rel, "kaggle")
        searched.append(f"{KAGGLE_INPUT}/<yolo dataset>/yolo_cv_best/weights/best.pt")
    rid = models_repo(repo)
    if hub != "off":
        try:
            present = YOLO_HUB_FILE in set(hub_list_files(rid, "model"))
        except Exception as e:
            present = False
            searched.append(f"Hub {rid}: unreachable ({_err(e)})")
        if present and hub == "probe":
            return Resolved(Path(f"hub://{rid}/{YOLO_HUB_FILE}"), "hub-available")
        if present:
            import shutil

            dest = Path(os.environ.get("GROUNDING_MODEL_PATH") or YOLO_DEFAULT_PATH)
            staging = Path("data/models/_hf_yolo_temp")
            got = Path(hub_file(rid, "model", YOLO_HUB_FILE, staging))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(got, dest)
            shutil.rmtree(staging, ignore_errors=True)
            return Resolved(dest, "hub")
        searched.append(f"Hub {rid}: {YOLO_HUB_FILE} not present")
    raise InputNotFound("YOLO grounding checkpoint (locate_tooth)", searched,
                        "python scripts/fetch_hf_assets.py --models   (or set GROUNDING_MODEL_PATH to your best.pt)")


# --------------------------------------------------------------------------------------------------------------------
# Base model
# --------------------------------------------------------------------------------------------------------------------
_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
KAGGLE_BASE_MODEL_DIRS = [
    "/kaggle/input/datasets/rezanadimikj/qwen3-5-9b", "/kaggle/input/qwen3-5-9b", "/kaggle/input/qwen3_5_9b",
    "/kaggle/input/qwen-3-5-9b", "/kaggle/input/qwen3-5-9b-base",
]


def hf_cached_model(model_id: str) -> Path | None:
    for root in _hf_cache_roots():
        for snap in sorted(root.glob(f"models--{model_id.replace('/', '--')}/snapshots/*")):
            if (snap / "config.json").is_file():
                return snap
    return None


def resolve_base_model(model_id: str, *, hub: HubMode = "off") -> Resolved:
    """Validate ``--model-id``: an existing local folder (config + weights) or a Hub id. A path that does not exist FAILS
    before any load. An explicit Hub id is never re-routed to a Kaggle copy (that choice belongs to the notebook)."""
    p = Path(model_id)
    if p.is_dir():
        has_cfg = (p / "config.json").is_file()
        has_w = any(x.suffix in (".safetensors", ".bin") for x in p.iterdir())
        if has_cfg and has_w:
            return Resolved(p, "local")
        raise InputNotFound(f"Base model folder {model_id}", [f"{p}: " + ("no config.json" if not has_cfg else "no weight files (.safetensors/.bin)")],
                            "attach the complete Kaggle model input or pass --model-id Qwen/Qwen3.5-9B (Hub)")
    if _HUB_ID.match(model_id) and not os.path.isabs(model_id) and not model_id.startswith("."):
        cached = hf_cached_model(model_id)
        if cached is not None:
            return Resolved(cached, "hub-cache")
        if hub == "probe":
            try:
                ok = "config.json" in set(hub_list_files(model_id, "model"))
            except Exception as e:
                raise InputNotFound(f"Base model {model_id}", [f"Hub {model_id}: unreachable or private ({_err(e)})"],
                                    "set HF_TOKEN, or pass --model-id <local folder>") from e
            if not ok:
                raise InputNotFound(f"Base model {model_id}", [f"Hub {model_id}: no config.json"], "check the model id")
        return Resolved(Path(model_id), "hub-available")
    raise InputNotFound(f"Base model {model_id}", [f"{model_id}: no such folder"] + [f"{d}" for d in KAGGLE_BASE_MODEL_DIRS],
                        "pass --model-id Qwen/Qwen3.5-9B (Hub) or the folder of the attached Kaggle model")
