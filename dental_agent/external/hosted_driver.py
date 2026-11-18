"""Drive a hosted (non-live) campaign from an interactive session: push shard tasks, schedule one run per model, watch, collect.

Everything is idempotent and recorded in ``<root>/campaign.json``, so the session can be stopped at any moment (a spend cap, a
closed laptop, a day or two off) and re-opened: ``push_missing`` / ``schedule`` / ``refresh`` / ``collect`` pick up exactly where
the state file says. Hosted runs write their answers where this machine can collect them (and, when configured, to the Hub);
nothing here ever needs a secret, and ground truth is only used afterwards, locally, by ``evaluate``.

A model run that is paused (cap, expired key) or killed keeps its finished images: ``collect`` merges whatever each shard
returned, and ``plan_remainders`` turns the images a model still lacks into small remainder tasks for that model only.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from dental_agent.external.conditions import get_condition
from dental_agent.external.identity import cohort_sha, run_settings
from dental_agent.external.importer import append_jsonl, cell_paths, completed, import_transport, read_jsonl
from dental_agent.external.kaggle_cli import CLIError, KaggleCLI
from dental_agent.external.model_profiles import HOSTED_DECODING, hosted_model
from dental_agent.external.persist import result_rows
from dental_agent.external.plugins import load_plugins
from dental_agent.external.sharding import ShardSpec, render_task_source, task_name

ACTIVE = {"scheduled", "running", "queued", "pending"}


def _slug_key(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


class HostedCampaign:
    def __init__(self, root: str | Path, cli: KaggleCLI, specs: list[ShardSpec], models: list[str], *,
                 log: Callable[[str], None] = print, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cli, self.log, self.clock = cli, log, clock
        self.models = list(models)
        self.state_path = self.root / "campaign.json"
        self.state: dict[str, Any] = {"version": 1, "tasks": {}, "runs": {}, "remainders": {}}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state.setdefault("remainders", {})
        self.specs: dict[str, ShardSpec] = {s.task_name: s for s in specs}
        load_plugins(sorted({p for s in specs for p in s.plugins}))        # the driver scores with the same plugins the workers ran
        self.task_models: dict[str, list[str]] = {name: list(models) for name in self.specs}
        self._adopt_remainders()

    def _adopt_remainders(self) -> None:
        for name, rem in self.state["remainders"].items():       # remainder tasks planned in an earlier session
            self.specs[name] = ShardSpec.from_json(rem["spec"])
            self.task_models[name] = list(rem["models"])

    # ------------------------------------------------------------------ state
    def _save(self) -> None:
        self.state["saves"] = int(self.state.get("saves", 0)) + 1
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    @staticmethod
    def _key(task: str, model: str) -> str:
        return f"{task}|{model}"

    def run_state(self, task: str, model: str) -> dict[str, Any] | None:
        return self.state["runs"].get(self._key(task, model))

    # ------------------------------------------------------------------ push
    def push_missing(self) -> dict[str, str]:
        """Push every shard task not yet on Kaggle at its current content. A real push also starts one run on the account's default
        model; it is adopted as that model's run (never scheduled twice)."""
        result: dict[str, str] = {}
        for name, spec in self.specs.items():
            known = self.state["tasks"].get(name)
            if known and known.get("spec_sha") == spec.sha() and not known.get("push_error"):
                result[name] = "up to date"
                continue
            path = self.root / "tasks" / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_task_source(spec), encoding="utf-8")
            try:
                version = self.cli.push(name, path, datasets=spec.attach)
            except CLIError as exc:
                self.state["tasks"][name] = {"spec_sha": spec.sha(), "push_error": str(exc)[:500]}
                self._save()
                self.log(f"[push] {name}: {exc}")
                result[name] = "error"
                continue
            for key in [k for k in self.state["runs"] if k.startswith(name + "|")]:
                del self.state["runs"][key]                      # a new version invalidates older runs of this task
            self.state["tasks"][name] = {"spec_sha": spec.sha(), "version": version, "pushed_at": self.clock()}
            try:
                for model, row in self.cli.status(name).rows.items():
                    self.state["runs"][self._key(name, model)] = {"status": row.status, "adopted_from_push": True}
            except CLIError:
                pass
            self._save()
            result[name] = f"pushed v{version}"
        return result

    # ------------------------------------------------------------------ schedule / refresh
    def inflight(self) -> int:
        return sum(1 for r in self.state["runs"].values() if r.get("status") in ACTIVE)

    def schedule(self, max_inflight: int = 60, only_models: list[str] | None = None, only_conditions: list[str] | None = None) -> dict[str, int]:
        n = {"scheduled": 0, "rejected": 0, "skipped_cap": 0}
        for name, spec in self.specs.items():
            if self.state["tasks"].get(name, {}).get("version") is None or self.state["tasks"][name].get("push_error"):
                continue
            if only_conditions and spec.condition not in only_conditions:
                continue
            for model in self.task_models[name]:
                if only_models and model not in only_models:
                    continue
                if self.run_state(name, model):
                    continue
                if self.inflight() >= max_inflight:
                    n["skipped_cap"] += 1
                    continue
                ok, text = self.cli.run_text(name, model)
                self.state["runs"][self._key(name, model)] = ({"status": "scheduled", "scheduled_at": self.clock()} if ok
                                                              else {"status": "rejected", "error": text})
                n["scheduled" if ok else "rejected"] += 1
                self._save()
        return n

    def refresh(self) -> dict[str, int]:
        """Update every task that still has an active run from ``tasks status``."""
        counts: dict[str, int] = {}
        for name in self.specs:
            keys = [k for k in self.state["runs"] if k.startswith(name + "|")]
            if not any(self.state["runs"][k].get("status") in ACTIVE for k in keys):
                continue
            try:
                status = self.cli.status(name)
            except CLIError as exc:
                self.log(f"[status] {name}: {exc}")
                continue
            for model, row in status.rows.items():
                key = self._key(name, model)
                run = self.state["runs"].get(key)
                if run is None or run.get("status") in ("rejected",) or run.get("collected"):
                    continue
                run["status"], run["started"], run["ended"] = row.status, row.started, row.ended
        self._save()
        for r in self.state["runs"].values():
            counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
        return counts

    def wait(self, poll_s: float = 30.0, timeout_s: float = 3600.0, sleep: Callable[[float], None] = time.sleep) -> dict[str, int]:
        deadline = self.clock() + timeout_s
        while True:
            counts = self.refresh()
            if not any(counts.get(s, 0) for s in ACTIVE) or self.clock() > deadline:
                return counts
            sleep(poll_s)

    # ------------------------------------------------------------------ collect
    def _run_dir(self, task_dir: Path, model: str) -> Path | None:
        """Latest ``<version>/<model>/<run_id>`` folder holding a manifest."""
        cands = []
        for mdir in task_dir.glob(f"*/{model}"):
            for rdir in mdir.iterdir():
                if (rdir / "shard_manifest.json").is_file() or (rdir / "transport.jsonl").is_file():
                    version = int(mdir.parent.name) if mdir.parent.name.isdigit() else 0
                    cands.append((version, int(rdir.name) if rdir.name.isdigit() else 0, rdir))
        return max(cands, key=lambda c: (c[0], c[1]))[2] if cands else None

    def collect(self, out_dir: str | Path) -> dict[str, int]:
        """Download finished runs and merge their transport records into ``out_dir`` (``<model>/<condition>.transport.jsonl``)."""
        out_dir = Path(out_dir)
        tally = {"collected": 0, "failed": 0, "no_output": 0}
        for name, spec in self.specs.items():
            todo = [m for m in self.task_models[name]
                    if (r := self.run_state(name, m)) and r.get("status") in ("completed", "failed", "errored", "error", "timeout", "cancelled", "canceled")
                    and not r.get("collected")]
            if not todo:
                continue
            try:
                task_dir = self.cli.download(name, self.root / "downloads")
            except CLIError as exc:
                self.log(f"[download] {name}: {exc}")
                continue
            for model in todo:
                run = self.state["runs"][self._key(name, model)]
                rdir = self._run_dir(Path(task_dir), model)
                if rdir is None:
                    run["collected"], run["outcome"] = True, "no_output"
                    tally["no_output"] += 1
                    continue
                manifest = json.loads((rdir / "shard_manifest.json").read_text()) if (rdir / "shard_manifest.json").is_file() else {}
                slug = manifest.get("model_raw_slug") or hosted_model(model).raw_slug
                paths = cell_paths(out_dir, slug, spec.condition)
                have = set(completed(read_jsonl(paths["transport"])))
                for rec in read_jsonl(rdir / "transport.jsonl"):
                    if rec.get("error") or int(rec["image_id"]) in have:
                        continue
                    append_jsonl(paths["transport"], rec)
                    have.add(int(rec["image_id"]))
                for kind in ("usage", "journal"):
                    for rec in read_jsonl(rdir / f"{kind}.jsonl"):
                        append_jsonl(paths[kind], rec)
                run.update(collected=True, outcome=manifest.get("status", "no_manifest"), model_raw_slug=slug,
                           manifest={k: manifest.get(k) for k in ("status", "stage", "error", "completed_images", "missing_images", "grounder_sha256",
                                                                  "settings_fingerprint", "python", "adaptations", "hub", "hub_error", "stats")},
                           repo_sha=(manifest.get("bootstrap") or {}).get("repo_sha"))
                tally["collected" if manifest.get("status") in ("complete", "paused", "incomplete") else "failed"] += 1
                self._save()
        return tally

    # ------------------------------------------------------------------ the Hub (an existing private repo; never created here)
    def _state_target(self, label: str) -> tuple[str, dict[str, Any]]:
        spec = next(iter(self.specs.values()))
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "default"
        prefix = f"external/campaign/{spec.dataset}/{spec.split}/{spec.cohort_sha256[:12]}/{safe}"
        return prefix, {"profile": safe, "kind": "campaign", "stage": "campaign", "cohort": spec.cohort_sha256, "split": spec.split}

    def sync_state_to_hub(self, hub, folder: str | Path, label: str = "default") -> None:
        """Keep ``campaign.json`` on the Hub: a new session (another day, another machine, a fresh Kaggle container) can then carry on
        instead of pushing and scheduling everything again. ``label`` separates Kaggle accounts (their tasks are separate)."""
        prefix, ident = self._state_target(label)
        hub.save_blob(folder, prefix, ident, "campaign.json", self.state_path.read_bytes(), int(self.state.get("saves", 0)))

    def restore_state_from_hub(self, hub, folder: str | Path, label: str = "default") -> bool:
        """Adopt the Hub's campaign state when this session has none of its own (never replaces local state). True when adopted."""
        if self.state["tasks"] or self.state["runs"]:
            return False
        prefix, ident = self._state_target(label)
        data = hub.load_blob(folder, prefix, ident, "campaign.json")
        if data is None:
            return False
        self.state = json.loads(data.decode("utf-8"))
        self.state.setdefault("remainders", {})
        self._adopt_remainders()
        self._save()
        return True

    def push_to_hub(self, hub, out_dir: str | Path, folder: str | Path) -> dict[str, int]:
        """Push every collected cell's finished transport records as the bundle ``collected`` (workers push their own shards; this
        covers runs that had no token and makes a second machine able to resume). Identity-checked and append-only on the Hub."""
        pushed: dict[str, int] = {}
        for condition in sorted({s.condition for s in self.specs.values()}):
            for model in self.models:
                rows = result_rows(cell_paths(out_dir, self.slug_of(model), condition)["transport"])
                if rows and hub.save(Path(folder) / "push", self.expected_settings(model, condition), rows, complete=False, shard="collected"):
                    pushed[f"{model}/{condition}"] = len(rows)
        return pushed

    def restore_from_hub(self, hub, out_dir: str | Path, folder: str | Path) -> dict[str, int]:
        """Pull every shard bundle of every cell into the local transport files (what a worker pushed, or an earlier session)."""
        added: dict[str, int] = {}
        for condition in sorted({s.condition for s in self.specs.values()}):
            for model in self.models:
                settings = self.expected_settings(model, condition)
                paths = cell_paths(out_dir, self.slug_of(model), condition)
                have = set(completed(read_jsonl(paths["transport"])))
                new = 0
                for i, rec in sorted(hub.restore_all_shards(Path(folder) / "pull", settings).items()):
                    if i not in have and i in set(self.cohort(condition)):
                        append_jsonl(paths["transport"], rec)
                        new += 1
                if new:
                    added[f"{model}/{condition}"] = new
        return added

    # ------------------------------------------------------------------ progress / repair
    def cohort(self, condition: str) -> list[int]:
        return sorted({i for n, s in self.specs.items() if s.condition == condition and n not in self.state["remainders"] for i in s.image_ids})

    def transport_ids(self, out_dir: str | Path, slug: str, condition: str) -> set[int]:
        return set(completed(read_jsonl(cell_paths(out_dir, slug, condition)["transport"])))

    def slug_of(self, model: str) -> str:
        for key, run in self.state["runs"].items():
            if key.endswith("|" + model) and run.get("model_raw_slug"):
                return run["model_raw_slug"]
        return hosted_model(model).raw_slug

    def missing(self, out_dir: str | Path, model: str, condition: str) -> list[int]:
        return sorted(set(self.cohort(condition)) - self.transport_ids(out_dir, self.slug_of(model), condition))

    def plan_remainders(self, out_dir: str | Path) -> list[str]:
        """For every model whose collected shards left images undone, add ONE remainder task (that model only). Returns new task names."""
        new: list[str] = []
        by_cond: dict[str, ShardSpec] = {}
        for n, s in self.specs.items():
            if n not in self.state["remainders"]:
                by_cond.setdefault(s.condition, s)
        for condition, template in by_cond.items():
            for model in self.models:
                runs = [r for k, r in self.state["runs"].items() if k.endswith("|" + model) and self.specs[k.split("|")[0]].condition == condition]
                if not runs or any(r.get("status") in ACTIVE or not r.get("collected") for r in runs):
                    continue                                  # still working, or nothing collected yet
                miss = self.missing(out_dir, model, condition)
                if not miss:
                    continue
                k = 1
                while task_name(template.cohort_sha256, condition, 99, remainder=(_slug_key(model), k)) in self.specs:
                    k += 1                                    # a model can need a second round after a second pause
                name = task_name(template.cohort_sha256, condition, 99, remainder=(_slug_key(model), k))
                spec = ShardSpec(**{**json.loads(template.to_json()), "task_name": name, "image_ids": miss,
                                    "rel_paths": {str(i): self._rel(condition, i) for i in miss}})
                self.specs[name] = spec
                self.task_models[name] = [model]
                self.state["remainders"][name] = {"spec": spec.to_json(), "models": [model]}
                new.append(name)
        self._save()
        return new

    def _rel(self, condition: str, image_id: int) -> str:
        for n, s in self.specs.items():
            if s.condition == condition and str(image_id) in s.rel_paths:
                return s.rel_paths[str(image_id)]
        raise KeyError(image_id)

    def paused_until(self) -> list[dict[str, Any]]:
        """Collected shards that stopped on a cap / expired key (with whatever refill time the provider gave)."""
        out = []
        for key, run in self.state["runs"].items():
            m = run.get("manifest") or {}
            paused = ((m.get("stats") or {}).get("paused")) if m else None
            if run.get("outcome") == "paused" and paused:
                out.append({"run": key, **paused})
        return out

    def report(self, out_dir: str | Path) -> dict[str, Any]:
        runs: dict[str, int] = {}
        for r in self.state["runs"].values():
            label = r.get("outcome") if r.get("collected") else r.get("status", "?")
            runs[label] = runs.get(label, 0) + 1
        cells = {}
        for condition in sorted({s.condition for s in self.specs.values()}):
            total = len(self.cohort(condition))
            for model in self.models:
                done = len(self.transport_ids(out_dir, self.slug_of(model), condition))
                cells[f"{model}/{condition}"] = f"{done}/{total}"
        return {"runs": runs, "cells": cells, "paused": self.paused_until(), "tasks": len(self.specs)}

    # ------------------------------------------------------------------ scoring
    def expected_settings(self, model: str, condition: str) -> dict[str, Any]:
        """The settings the hosted run must have used, rebuilt locally. The grounder hash is taken from the runs themselves (they must
        all agree); prompt / harness hashes come from THIS checkout, so a hosted clone that differs from it is caught on import."""
        spec = next(s for s in self.specs.values() if s.condition == condition)
        shas = {(r.get("manifest") or {}).get("grounder_sha256") for k, r in self.state["runs"].items()
                if self.specs[k.split("|")[0]].condition == condition and r.get("manifest")}
        shas.discard(None)
        if len(shas) > 1:
            raise ValueError(f"hosted runs of {condition} used different grounder weights: {sorted(shas)}")
        return run_settings(get_condition(condition), self.slug_of(model), cohort=spec.cohort_sha256, split=spec.split, dataset=spec.dataset,
                            grounder_sha256=next(iter(shas), None), overrides=spec.overrides, decoding=HOSTED_DECODING,
                            cohort_settings=spec.cohort_settings)

    def evaluate(self, out_dir: str | Path, annots_df, cats_df, models: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Score every collected cell locally (ground truth never left this machine). Returns per-cell import results."""
        results: dict[str, dict[str, Any]] = {}
        for condition in sorted({s.condition for s in self.specs.values()}):
            spec = next(s for s in self.specs.values() if s.condition == condition)
            for model in models or self.models:
                slug = self.slug_of(model)
                paths = cell_paths(out_dir, slug, condition)
                if not paths["transport"].exists():
                    continue
                results[f"{model}/{condition}"] = import_transport(
                    paths["transport"], annots_df, cats_df, settings=self.expected_settings(model, condition), dataset=spec.dataset,
                    split=spec.split, cohort_ids=set(self.cohort(condition)), evaluated_path=paths["evaluated"])
        return results


def build_campaign(root: str | Path, images_df, *, conditions: list[str], models: list[str] | None = None, split: str = "test",
                   dataset: str = "dentex", image_ids=None, shard_size: int = 25, workers: int = 4, env_dataset: str | None = "auto",
                   hf_results_repo: str | None = None, hf_repo_type: str | None = None, plugins: list[str] | None = None,
                   extra_datasets: list[str] | None = None, image_datasets: list[str] | None = None, dataset_dirs=None,
                   overrides: dict[str, Any] | None = None, cli: KaggleCLI | None = None, log: Callable[[str], None] = print,
                   cohort_settings: dict[str, Any] | None = None, **spec_kw) -> HostedCampaign:
    """A campaign over ``images_df`` (the table the notebook / script already resolved) for these conditions and models.

    * the image datasets are READ BACK from where the images are mounted (``locate_image_datasets``), nothing is typed;
    * ``env_dataset="auto"`` attaches the attached private dataset that holds a ``.env`` (``config.discover_env_datasets``);
    * ``cohort_settings`` is what ``cohort.load_split`` returned for the 250-image cohort (required there, refused elsewhere);
    * a tool condition needs the YOLO weights on the worker: either the env dataset (HF_TOKEN reaches the private models repo) or a
      weights dataset in ``extra_datasets``, and this fails HERE, before anything is pushed, when neither is given.
    """
    from dental_agent.config import discover_env_datasets
    from dental_agent.external.model_profiles import hosted_vision_models
    from dental_agent.external.sharding import locate_image_datasets, specs_for_conditions
    from dental_agent.external.identity import cohort_sha
    from dental_agent.external.plugins import load_plugins as _load
    _load(plugins or [])
    from dental_agent.external.cohort import check_settings
    check_settings(split, cohort_settings)
    ids = sorted(int(i) for i in (image_ids if image_ids is not None else images_df["id"]))
    if dataset_dirs is None:
        from dental_agent.utils.inputs import kaggle_dataset_dirs
        dataset_dirs = kaggle_dataset_dirs()
    if env_dataset == "auto":
        found = discover_env_datasets()
        env_dataset = found[0][0] if found else None
    located, rel = locate_image_datasets(images_df, ids, dataset_dirs)
    if any(get_condition(c).with_tools for c in conditions) and not (env_dataset or extra_datasets):
        raise ValueError("tool conditions need the YOLO weights on the worker: attach the private .env dataset (its HF_TOKEN reaches the "
                         "models repo) or pass the weights dataset in extra_datasets")
    specs = specs_for_conditions(conditions, image_ids=ids, rel_paths=rel, image_datasets=image_datasets or located,
                                 cohort_sha256=cohort_sha(ids, cohort_settings), split=split, cohort_settings=cohort_settings, dataset=dataset, shard_size=shard_size, workers=workers,
                                 env_dataset=env_dataset, hf_results_repo=hf_results_repo, hf_repo_type=hf_repo_type,
                                 plugins=plugins, extra_datasets=extra_datasets, overrides=overrides, **spec_kw)
    return HostedCampaign(root, cli or KaggleCLI(), specs, models or hosted_vision_models(), log=log)
