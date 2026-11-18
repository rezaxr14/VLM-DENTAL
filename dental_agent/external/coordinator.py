"""Run one (model, condition) cell over a cohort with the project's own loop, and write transport records.

Per image: a fresh ``KaggleReply`` (one chat per image), the unmodified ``run_agent`` / ``run_agent_no_tools``, a transport
record with the answers and per-request usage. Every model turn is journalled the moment it arrives (``journal.py``), so an
image interrupted half-way resumes without paying for its answered turns again.

Stopping, in order of preference:

* ``auth`` / ``quota`` provider failures, or a ``QuotaGuard`` reading, PAUSE the run (``stats['paused']``). Nothing is recorded
  as an error for the image in flight; resume later (after the key is renewed or the allowance refills) and carry on.
* ``rate_limit`` is retried for the same image after a back-off (the journal makes the retry free) and pauses when it persists.
* any other infrastructure error is stored as evidence and the image is re-run on resume; ``max_consecutive_errors`` of them
  in a row stop the cell (a broken endpoint would otherwise fail every image).
* a spend budget stops the cell at an image boundary.

Images are independent, so ``workers > 1`` runs several at once (the proxy accepts 16 parallel requests).
"""
from __future__ import annotations

import functools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from dental_agent.external import records as R
from dental_agent.external.conditions import Condition, LOOP_DEFAULTS, get_condition
from dental_agent.external.cost_guard import BudgetExceeded, CostLedger
from dental_agent.external.identity import assert_same, fingerprint
from dental_agent.external.importer import append_jsonl, cell_paths, completed, read_jsonl
from dental_agent.external.journal import TurnJournal
from dental_agent.external.kaggle_provider import KaggleReply, ProviderError
from dental_agent.external.pause import RunPaused, classify
from dental_agent.external.transcript_sync import HistoryMismatch

RATE_LIMIT_BACKOFF_S = (30.0, 60.0, 120.0)


def _run_image(cond: Condition, image_id: int, images_df, reply, cfg: dict[str, Any], registry, prompts=None):
    """``prompts`` (``cohort.Prompts``): the prompt texts of the run's split; ``None`` = the base prompts."""
    from dental_agent.external.cohort import prompts_for
    prompts = prompts or prompts_for(None)
    from dental_agent.utils.canonical import set_letterbox
    set_letterbox(False)                                 # GPU-profile views: no black bars
    if cond.prompt == "legacy_zero_shot":
        from PIL import Image
        row = images_df[images_df["id"] == image_id].iloc[0]
        from dental_agent.utils.canonical import to_canonical
        image = to_canonical(Image.open(row["local_path"]).convert("RGB"), "FULL")
        messages = [{"role": "system", "content": R.LEGACY_SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompts.zero_shot}]}]
        text = reply(None, None, messages, max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"], return_ids=True,
                     past_key_values=None, cache_state=None, return_cache=True)[0]
        return R.legacy_zero_shot_trajectory(image_id, text)
    if not cond.with_tools:
        from dental_agent.agent.loop import run_agent_no_tools
        return run_agent_no_tools(image_id, images_df, None, None, verbose=False, canonical_resize=True,
                                  temperature=cfg["temperature"], max_new_tokens=cfg["max_new_tokens"], system_prompt=prompts.no_tools, reply_fn=reply)
    from dental_agent.agent.loop import run_agent
    from dental_agent.harness.configuration import harness_options
    return run_agent(image_id, images_df, None, None, registry=registry, max_tool_calls=cfg["max_tool_calls"], verbose=False,
                     canonical_resize=True, temperature=cfg["temperature"], max_new_tokens=cfg["max_new_tokens"], dataset=prompts.agent_dataset,
                     malformed_retries=cfg["malformed_retries"], harness=harness_options(cond.harness_mode), reply_fn=reply)


def run_cell(*, backend, condition: str, images_df, image_ids: list[int], out_dir, settings: dict[str, Any],
             ledger: CostLedger | None = None, max_consecutive_errors: int = 3, registry=None,
             run_image: Callable[..., Any] | None = None, overrides: dict[str, Any] | None = None,
             log: Callable[[str], None] = print, guard=None, workers: int = 1, journal: bool = True,
             sleep: Callable[[float], None] = time.sleep, rate_limit_backoff: tuple[float, ...] = RATE_LIMIT_BACKOFF_S) -> dict[str, Any]:
    cond = get_condition(condition)
    cfg = {**LOOP_DEFAULTS, **(overrides or {})}
    if run_image is None:                                # the default image runner shows the model the prompts of the run's split
        from dental_agent.external.cohort import prompts_for
        run_image = functools.partial(_run_image, prompts=prompts_for(settings.get("split"), settings.get("dataset", "dentex")))
    paths = cell_paths(out_dir, backend.model_slug, condition)
    existing = read_jsonl(paths["transport"])
    done = completed(existing)
    for rec in done.values():                            # a resume must match the saved identity exactly
        assert_same(rec.get("identity"), settings)
    if cond.with_tools and registry is None:
        from dental_agent.external.tool_packs import get_registry
        registry = get_registry(cond.pack)
    ident_fp = fingerprint(settings)
    turns = TurnJournal(paths["journal"]) if journal else None
    stats: dict[str, Any] = {"condition": condition, "model": backend.model_slug, "done": len(done), "ran": 0, "errors": 0,
                             "replayed_turns": 0, "stopped": None, "paused": None}
    pending = [int(i) for i in image_ids if int(i) not in done]
    lock = threading.Lock()
    stop = threading.Event()
    state = {"consecutive": 0}

    def halt(reason: str, paused: dict[str, Any] | None = None) -> None:
        with lock:
            if stats["stopped"] is None:
                stats["stopped"] = reason
                stats["paused"] = paused
        stop.set()

    def one(image_id: int) -> None:
        if stop.is_set():
            return
        try:
            if guard is not None:
                guard.check()
            if ledger is not None:
                ledger.check()
        except RunPaused as exc:
            return halt(f"paused: {exc.kind}: {exc}", {"kind": exc.kind, "message": str(exc), "refill_at": exc.refill_at})
        except BudgetExceeded as exc:
            return halt(f"budget: {exc}")
        started = time.time()
        attempt = 0
        while True:
            usage: list[dict[str, Any]] = []

            def sink(u, usage=usage):
                usage.append(u)
                if ledger is not None:
                    ledger.record(u)
                append_jsonl(paths["usage"], u)

            reply = KaggleReply(backend, image_id=image_id, usage_sink=sink, journal=turns)
            try:
                traj = run_image(cond, image_id, images_df, reply, cfg, registry)
                rec = R.transport_record(image_id=image_id, condition=condition, model_slug=backend.model_slug, traj=traj,
                                         usage=reply.usage or usage, seconds=time.time() - started, identity=settings)
                with lock:
                    state["consecutive"] = 0
                    stats["ran"] += 1
                    stats["replayed_turns"] += sum(1 for u in reply.usage if u.get("replayed"))
                append_jsonl(paths["transport"], rec)
                return
            except (ProviderError, HistoryMismatch) as exc:
                kind = classify(exc) if isinstance(exc, ProviderError) else "history_mismatch"
                if getattr(exc, "kind", None) == "config":
                    raise
                if kind in ("auth", "quota"):
                    return halt(f"paused: {kind}: {exc}", {"kind": kind, "message": str(exc)[:300], "refill_at": None})
                if kind == "rate_limit":
                    if attempt < len(rate_limit_backoff) and not stop.is_set():
                        log(f"[{backend.model_slug}/{condition}] image {image_id}: rate limited, waiting {rate_limit_backoff[attempt]:.0f}s")
                        sleep(rate_limit_backoff[attempt])
                        attempt += 1
                        continue                          # the journal replays answered turns; only the failed turn is re-sent
                    return halt(f"paused: rate_limit: {exc}", {"kind": "rate_limit", "message": str(exc)[:300], "refill_at": None})
                rec = {"kind": "transport", "image_id": image_id, "condition": condition, "model_slug": backend.model_slug,
                       "identity": settings, "usage": reply.usage or usage, "seconds": time.time() - started, "timestamp": time.time(),
                       "error": {"type": type(exc).__name__, "kind": getattr(exc, "kind", "history_mismatch"), "message": str(exc)[:500]}}
                with lock:
                    stats["errors"] += 1
                    state["consecutive"] += 1
                    tripped = state["consecutive"] >= max_consecutive_errors
                    n = state["consecutive"]
                log(f"[{backend.model_slug}/{condition}] image {image_id}: {rec['error']['type']}: {rec['error']['message']}")
                append_jsonl(paths["transport"], rec)
                if tripped:
                    halt(f"{n} consecutive errors")
                return

    if workers <= 1:
        for image_id in pending:
            one(image_id)
            if stop.is_set():
                break
    elif pending:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            list(pool.map(one, pending))
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    paths["meta"].write_text(json.dumps({"settings": settings, "fingerprint": ident_fp, "stats": stats}, indent=2, default=str))
    return stats
