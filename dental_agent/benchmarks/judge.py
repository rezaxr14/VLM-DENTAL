"""LLM judge for the open-ended answers: a generic OpenAI-compatible client plus a resumable, concurrent judging loop.

The judge model, endpoint and key variable are settings (never hardcoded). The prompt and the 5-try rule are the official ones
(``mmoral_opg.judge_one``). Every judged row stores the judge identity; a result file is only resumed with the same identity.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from dental_agent.benchmarks import mmoral_opg as mm

TRANSIENT = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "APIStatusError")


class JudgeClient:
    """``complete(prompt, temperature) -> str``. Transport errors are retried with backoff, then raised (never scored as 0.0)."""

    def __init__(self, model: str, *, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 max_completion_tokens: int = 16384, send_temperature: bool = True, transport_retries: int = 6,
                 client: Any = None, sleep: Callable[[float], None] = time.sleep):
        if not model or not isinstance(model, str):
            raise ValueError("Set the judge model explicitly (JUDGE_MODEL); there is no default.")
        self.model, self.base_url, self.api_key_env = model, base_url, api_key_env
        self.max_completion_tokens, self.send_temperature = int(max_completion_tokens), bool(send_temperature)
        self.transport_retries, self._sleep = int(transport_retries), sleep
        if client is None:
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"{api_key_env} is not set: the judge cannot run.")
            from openai import OpenAI
            client = OpenAI(api_key=key, **({"base_url": base_url} if base_url else {}))
        self.client = client

    def identity(self) -> dict[str, Any]:
        return dict(judge_model=self.model, judge_base_url=self.base_url or "", max_completion_tokens=self.max_completion_tokens,
                    send_temperature=self.send_temperature, judge_prompt_sha256=mm.JUDGE_PROMPT_SHA256, judge_tries=mm.JUDGE_TRIES)

    def complete(self, prompt: str, temperature: float) -> str:
        kwargs: dict[str, Any] = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                                      max_completion_tokens=self.max_completion_tokens)
        if self.send_temperature:
            kwargs["temperature"] = temperature
        for attempt in range(self.transport_retries + 1):
            try:
                reply = self.client.chat.completions.create(**kwargs)
                return reply.choices[0].message.content or ""
            except Exception as error:                                   # noqa: BLE001 - classified below
                name = type(error).__name__
                status = getattr(error, "status_code", None)
                transient = name in TRANSIENT and (status is None or status == 429 or status >= 500)
                if status == 400 and self.send_temperature and "temperature" in str(error).lower():
                    raise RuntimeError("The judge rejected the temperature parameter. Set JUDGE_SEND_TEMPERATURE = False (the "
                                       "judge then runs at its own default, which is recorded in the result identity).") from error
                if not transient or attempt == self.transport_retries:
                    raise
                self._sleep(min(60.0, 2.0 ** attempt))
        raise AssertionError("unreachable")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_answers(answers: list[dict[str, Any]], out_path: str | Path, client: JudgeClient, *, workers: int = 4,
                  existing: str = "resume", progress_every: int = 25) -> list[dict[str, Any]]:
    """Judge ``answers`` (dicts with ``index``, ``question``, ``ground_truth``, ``prediction``, ``category``).

    Appends one record per question to ``out_path``. ``existing``: ``error`` (refuse a non-empty file), ``resume`` (identity must match),
    ``overwrite``. Returns all records (old and new) for the answers given.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    identity = client.identity()
    old = read_jsonl(out_path)
    if old and existing == "overwrite":
        out_path.unlink()
        old = []
    elif old and existing != "resume":
        raise SystemExit(f"{out_path} already has {len(old)} judged rows. Use existing='resume' or 'overwrite'.")
    for rec in old:
        diff = {k: (rec.get(k), v) for k, v in identity.items() if rec.get(k) != v}
        if diff:
            raise SystemExit(f"{out_path} was judged with different settings {diff}; mixing them would make the score meaningless. "
                             "Use a different judge file or existing='overwrite'.")
    done = {rec["index"]: rec for rec in old}
    todo = [a for a in answers if a["index"] not in done]
    print(f"[JUDGE] {len(done)} already judged, {len(todo)} to judge with {identity['judge_model']}.", flush=True)

    def one(a):
        if a.get("refused"):                 # the generator produced no answer (TPU static generation): 0.0, no API call, flagged in the log
            res = dict(score=0.0, log="generation refused: no answer, scored 0.0 without calling the judge", tries=0)
            return dict(index=a["index"], category=a["category"], **res, **identity)
        res = mm.judge_one(client.complete, a["question"], a["ground_truth"], a["prediction"])
        return dict(index=a["index"], category=a["category"], **res, **identity)

    with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool, open(out_path, "a", encoding="utf-8") as sink:
        futures = {pool.submit(one, a): a["index"] for a in todo}
        try:
            for n, fut in enumerate(cf.as_completed(futures), 1):
                rec = fut.result()                 # a transport error here stops the run; finished rows are already on disk
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                done[rec["index"]] = rec
                if progress_every and n % progress_every == 0:
                    print(f"[JUDGE] {n}/{len(todo)}", flush=True)
        except BaseException:
            for fut in futures:
                fut.cancel()
            raise
    wanted = {a["index"] for a in answers}
    return [done[i] for i in sorted(wanted)]
