"""The 250-image DENTEX test cohort (``docs/OFFICIAL_TEST250.md``) as the external-model component uses it.

Nothing about the cohort is re-implemented here. This module only decides, from a split name, which of the project's own pieces to
call, so that every external condition reads the same data, shows the model the same cohort prompts and is scored the same way as the
trained checkpoint on that split:

* data      ``harness.notebook_eval.prepare_inputs`` (local disk, attached Kaggle dataset, Hugging Face cache, ``DENTEX_TEST_REPO``)
            then ``scripts.evaluate_models.load_eval_split``, which puts the cohort settings on ``args.test250_settings``;
* prompts   ``official_test250.zero_shot_prompt / no_tools_prompt`` and the agent prompt's ``official_test250`` dataset key;
* scoring   ``evaluation.test250_scoring.score_cohort`` (in ``records.evaluated_record``, keyed by the settings below);
* identity  the settings (COCO and mapping versions/hashes, scoring profile, prompt classes) are part of every run's identity
            under the same ``official_test250`` key ``evaluate_condition`` uses.

Every other split keeps the base prompts and has no cohort settings, so its identities and records are unchanged.
"""
from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

from dental_agent.data.official_test250 import SPLIT as COHORT_SPLIT

SETTINGS_KEY = "official_test250"       # key of the cohort settings in eval_settings / run_settings (as in scripts/evaluate_models.py)


@dataclass(frozen=True)
class Prompts:
    agent_dataset: str                  # ``dataset=`` of ``build_agent_system_prompt`` (selects the class list of the tool prompt)
    no_tools: str                       # system prompt of the no-tools condition
    zero_shot: str                      # user prompt of the legacy zero-shot supplement


def prompts_for(split: str | None, dataset: str = "dentex") -> Prompts:
    from dental_agent.agent import prompts as P
    if split == COHORT_SPLIT:
        from dental_agent.data import official_test250 as o
        return Prompts(COHORT_SPLIT, o.no_tools_prompt(), o.zero_shot_prompt())
    return Prompts(dataset, P.NO_TOOLS_SYSTEM_PROMPT, P.ZERO_SHOT_PROMPT)


def check_settings(split: str | None, cohort_settings: dict[str, Any] | None) -> None:
    """A cohort run without its settings would be scored with the 4-class scorer and carry no COCO identity; any other split with them
    would claim an identity it does not have. Fail closed in both directions."""
    if (split == COHORT_SPLIT) != bool(cohort_settings):
        raise ValueError(f"split {split!r} " + ("needs" if split == COHORT_SPLIT else "must not have") + f" the {SETTINGS_KEY} settings "
                         "(coco/mapping versions and hashes); load the split with external.cohort.load_split")


def scores_as_cohort(settings: dict[str, Any] | None) -> bool:
    return bool((settings or {}).get(SETTINGS_KEY))


def load_split(split: str, *, dataset: str = "dentex", data_dir: str = "data", dev_split: str | None = None, token: str | None = None,
               api: Any = None, download: Any = None):
    """``(images_df, annots_df, cats_df, cohort_settings)`` for ``split``. ``cohort_settings`` is ``None`` for every split but the cohort.
    ``api`` / ``download`` (a ``HfApi`` and ``hf_hub_download`` by default) are only used for the cohort, to resolve its COCO and images."""
    from scripts import evaluate_models as ev
    args = Namespace(split=split, dataset=dataset, dev_split=dev_split, data_dir=data_dir)
    if split == COHORT_SPLIT:
        from dental_agent.harness.notebook_eval import prepare_inputs
        if api is None or download is None:
            from huggingface_hub import HfApi, hf_hub_download
            api, download = api or HfApi(token=token), download or hf_hub_download
        token = token or os.environ.get("HF_TOKEN") or None
        args.data_dir = prepare_inputs(data_dir, COHORT_SPLIT, None, api, download, token)["data_dir"]
    images, annots, cats = ev.load_eval_split(args)
    cohort = getattr(args, "test250_settings", None)
    check_settings(split, cohort)
    return images, annots, cats, cohort
