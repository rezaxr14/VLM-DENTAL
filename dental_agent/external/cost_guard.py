"""Budget estimate and a spend ledger for hosted-model runs.

The loop re-sends the whole transcript every turn (or the provider bills the whole chat), so input tokens grow roughly
quadratically with the number of turns. ``worst_case`` uses the loop's hard bound (50 tool calls -> 65 turns); ``expected``
uses a measured mean from a pilot. Neither is a quote: Kaggle's pricing and quota semantics are gate G1 items, so a price is
optional and unknown usage is reported as unknown (None), never as 0.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dental_agent.external.conditions import LOOP_DEFAULTS, get_condition, max_turns


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


def requests_for(n_images: int, turns_per_image: float) -> float:
    return n_images * turns_per_image


def input_tokens_for(turns: float, base_tokens: int, added_per_turn: int) -> float:
    """Sum over turns t=1..T of (base + (t-1)*added): the history is re-sent each turn."""
    t = float(turns)
    return t * base_tokens + added_per_turn * t * (t - 1) / 2.0


def estimate(condition: str, n_images: int, *, base_tokens: int = 2500, added_per_turn: int = 900, out_tokens_per_turn: int = 400,
             expected_turns: float | None = None, price: Price | None = None, max_tool_calls: int = LOOP_DEFAULTS["max_tool_calls"]) -> dict[str, Any]:
    """Requests / tokens / cost for one model and one condition. ``worst`` assumes every image runs the full turn bound."""
    cond = get_condition(condition)
    worst_turns = max_turns(max_tool_calls) if cond.with_tools else 1
    exp_turns = float(expected_turns) if expected_turns is not None else (min(12.0, worst_turns) if cond.with_tools else 1.0)

    def block(turns: float) -> dict[str, Any]:
        inp = n_images * input_tokens_for(turns, base_tokens, added_per_turn)
        out = n_images * turns * out_tokens_per_turn
        cost = None if price is None else inp / 1e6 * price.input_per_mtok + out / 1e6 * price.output_per_mtok
        return {"turns_per_image": turns, "requests": requests_for(n_images, turns), "input_tokens": inp, "output_tokens": out, "cost_usd": cost}

    return {"condition": condition, "n_images": n_images, "assumptions": {"base_tokens": base_tokens, "added_per_turn": added_per_turn,
            "out_tokens_per_turn": out_tokens_per_turn, "priced": price is not None}, "expected": block(exp_turns), "worst": block(float(worst_turns))}


@dataclass
class Budget:
    max_requests: int | None = None
    max_cost_usd: float | None = None
    max_input_tokens: int | None = None


class CostLedger:
    """Append-only JSONL ledger. ``check`` runs BEFORE each image, so a budget stops the run at an image boundary."""

    def __init__(self, path: str | Path | None, budget: Budget | None = None):
        self.path = Path(path) if path else None
        self.budget = budget or Budget()
        self.requests = 0
        self.input_tokens = 0
        self.cost = 0.0
        self.unknown_usage = 0
        self.unknown_cost = 0
        self._lock = threading.Lock()            # worker threads record usage concurrently
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._add(json.loads(line))

    def _add(self, u: dict[str, Any]) -> None:
        self.requests += 1
        if u.get("prompt_tokens") is None:
            self.unknown_usage += 1
        else:
            self.input_tokens += int(u["prompt_tokens"])
        if u.get("cost_usd") is None:
            self.unknown_cost += 1
        else:
            self.cost += float(u["cost_usd"])

    def record(self, usage: dict[str, Any]) -> None:
        with self._lock:
            self._add(usage)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(usage, default=str) + "\n")

    def check(self) -> None:
        b = self.budget
        if b.max_requests is not None and self.requests >= b.max_requests:
            raise BudgetExceeded(f"request budget reached ({self.requests}/{b.max_requests})")
        if b.max_input_tokens is not None and self.input_tokens >= b.max_input_tokens:
            raise BudgetExceeded(f"input-token budget reached ({self.input_tokens}/{b.max_input_tokens})")
        if b.max_cost_usd is not None:
            if self.unknown_cost and self.cost == 0.0 and self.requests == self.unknown_cost:
                return      # the provider reports no cost: only the request / token limits can stop the run
            if self.cost >= b.max_cost_usd:
                raise BudgetExceeded(f"cost budget reached (${self.cost:.2f}/${b.max_cost_usd:.2f})")

    def summary(self) -> dict[str, Any]:
        return {"requests": self.requests, "input_tokens": self.input_tokens if not self.unknown_usage else None,
                "known_input_tokens": self.input_tokens, "cost_usd": self.cost if not self.unknown_cost else None,
                "known_cost_usd": self.cost, "n_unknown_usage": self.unknown_usage, "n_unknown_cost": self.unknown_cost}
