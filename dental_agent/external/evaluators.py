"""Evaluators: how a transport record (answers + usage, no ground truth) becomes an evaluated record, locally.

``dentex_findings`` is the project's own scoring (``records.evaluated_record``, the exact schema ``evaluate_condition`` writes, so
the matrix / bootstrap / paper code reads it unchanged). A different task (segmentation masks, a retrieval-augmented condition)
registers its own evaluator and names it in ``Condition(evaluator=...)``; the importer, the hosted driver and the tables do not change.
Signature: ``evaluator(transport_record, annots_df, cats_df, *, dataset, split, settings, model_slug) -> dict``.
"""
from __future__ import annotations

from typing import Any, Callable

Evaluator = Callable[..., dict[str, Any]]
_EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(name: str, fn: Evaluator, replace: bool = False) -> None:
    if name in _EVALUATORS and not replace:
        raise ValueError(f"evaluator {name!r} is already registered")
    _EVALUATORS[name] = fn


def get_evaluator(name: str = "dentex_findings") -> Evaluator:
    try:
        return _EVALUATORS[name]
    except KeyError:
        raise ValueError(f"Unknown evaluator '{name}'. Known: {', '.join(_EVALUATORS)}") from None


def _dentex_findings(rec, annots_df, cats_df, *, dataset, split, settings, model_slug):
    from dental_agent.external.records import evaluated_record
    return evaluated_record(rec, annots_df, cats_df, dataset=dataset, split=split, settings=settings, model_slug=model_slug)


_EVALUATORS["dentex_findings"] = _dentex_findings
