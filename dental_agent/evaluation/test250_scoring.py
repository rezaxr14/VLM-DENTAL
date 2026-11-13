"""Scoring of the 250-image test cohort, on top of ``match_multi_findings`` (never instead of it).

Every polygon code is a class (see ``dental_agent/data/official_test250.py``); nothing is dropped. The only things done here:

* canonical class names: a model may write "Root Canal Treatment", "tooth extraction", "fractured tooth", ... . The shared normaliser
  already maps everything it knows (Caries, Deep Caries, Periapical Lesion with every "lesion" name, Impacted) and leaves the rest as
  free text; this puts that remainder into the four cohort names (Root Canal, Curettage, Extraction, Fracture). Names it cannot place are
  left alone and simply do not match;
* ``family_exact_*``: exact-match precision/recall/F1 with Caries and Deep Caries treated as one class, because the labels do not
  separate them. The standard metrics keep their own proximity credit (Caries <-> Deep Caries 0.75, localisation F1, closeness score).

Dev, training and the 50-image split never reach this module.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from dental_agent.evaluation.metrics import match_multi_findings, normalize_dental_diagnosis

PROFILE = "test250_scoring_v2"
KNOWN = {"Caries", "Deep Caries", "Periapical Lesion", "Impacted"}
# (canonical class, substrings of the lower-cased free text). Only reached when the shared normaliser did not place the text.
PATTERNS = (
    ("Root Canal", ("root canal", "endodontic", "root filling", "pulpectomy", "rct")),
    ("Curettage", ("curettage", "curetage", "curettement", "kuretaj")),
    ("Extraction", ("extraction", "extracted", "exodontia")),
    ("Fracture", ("fracture", "fractured", "broken", "crack")),
)


def canonical_class(finding: Mapping[str, Any]) -> str:
    """The class name this cohort scores a prediction under."""
    shared = normalize_dental_diagnosis(finding.get("diagnosis"))
    if shared in KNOWN:
        return shared
    text = f"{finding.get('raw_diagnosis') or ''} {finding.get('diagnosis') or ''}".lower()
    for name, needles in PATTERNS:
        if any(re.search(r"\b" + re.escape(n), text) for n in needles):
            return name
    return shared


def family(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copies with Deep Caries folded into Caries (used for the family-level metric only)."""
    out = []
    for f in findings:
        g = dict(f)
        if normalize_dental_diagnosis(g.get("diagnosis")) == "Deep Caries":
            g["diagnosis"] = "Caries"
        out.append(g)
    return out


def score_cohort(gt: Sequence[Mapping[str, Any]], preds: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (predictions with canonical class names, extra record fields). The caller runs the standard ``match_multi_findings`` on them."""
    canon = []
    renamed = 0
    for p in preds:
        q = dict(p)
        name = canonical_class(p)
        if name != normalize_dental_diagnosis(p.get("diagnosis")):
            renamed += 1
        q["diagnosis"] = name
        canon.append(q)
    fam = match_multi_findings(family(gt), family(canon))
    return canon, dict(scoring_profile=PROFILE, n_renamed_predictions=renamed, family_exact_precision=fam["exact_precision"],
                       family_exact_recall=fam["exact_recall"], family_exact_f1=fam["exact_f1"])
