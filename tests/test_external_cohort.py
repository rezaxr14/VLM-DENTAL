"""The external-model component on the 250-image cohort (docs/OFFICIAL_TEST250.md): the same data loader, prompts, scoring and identity as the
trained checkpoint's evaluation, and no change at all for every other split. Offline, scripted fake backend, no torch."""
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from dental_agent.data import official_test250 as o
from dental_agent.external import cohort as K
from dental_agent.external import conditions as C
from dental_agent.external import coordinator, identity, importer, preflight, records
from dental_agent.external.sharding import build_specs
from tests.test_external_kaggle import EMPTY, FINAL, FakeBackend

SETTINGS = dict(coco_version="v1", mapping_version="v1", mapping_sha256="m" * 64, source_zip_sha256=None, scoring="test250_scoring_v2",
                prompt_classes=list(o.PROMPT_CLASSES))
ROOT_CANAL = json.dumps({"thought": "x", "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Root Canal Treatment", "confidence": 0.9}]})


def settings(cond, ids=(100, 101, 102), split=o.SPLIT, cohort=SETTINGS, slug="fake/model-1"):
    return identity.run_settings(C.get_condition(cond), slug, cohort=identity.cohort_sha(ids, cohort), split=split, grounder_sha256="g" * 64,
                                 cohort_settings=cohort)


@pytest.fixture
def table(tmp_path):
    from PIL import Image
    rows = []
    for i in (100, 101, 102):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 48), (i, 10, 10)).save(p)
        rows.append({"id": i, "local_path": str(p)})
    annots = pd.DataFrame([{"id": 1, "image_id": 100, "category_id_1": 0, "category_id_2": 5, "category_id_3": 5, "bbox": [10, 10, 10, 10]}])
    cats = pd.DataFrame([{"id": k + 1, "name": n} for k, n in enumerate(["Caries", "Deep Caries", "Periapical Lesion", "Impacted", "Root Canal", "Curettage", "Extraction", "Fracture"])])
    return pd.DataFrame(rows), annots, cats


# ------------------------------------------------------------------------------------------------------------------ prompts
def test_other_splits_keep_the_base_prompts_and_the_cohort_gets_the_derived_ones():
    from dental_agent.agent import prompts as P
    base = K.prompts_for("test")
    assert (base.agent_dataset, base.no_tools, base.zero_shot) == ("dentex", P.NO_TOOLS_SYSTEM_PROMPT, P.ZERO_SHOT_PROMPT)
    assert K.prompts_for("dev", "tufts").agent_dataset == "tufts"
    cohort = K.prompts_for(o.SPLIT)
    assert cohort.agent_dataset == o.SPLIT and cohort.no_tools == o.no_tools_prompt() and cohort.zero_shot == o.zero_shot_prompt()
    assert "Root Canal" in cohort.no_tools and "Root Canal" not in base.no_tools


def test_prompt_hashes_follow_the_split_and_base_hashes_do_not_move():
    for name in ("kaggle_zero_shot", "kaggle_tools", "kaggle_zero_shot_legacy_prompt"):
        cond = C.get_condition(name)
        assert identity.prompt_hashes(cond, split="test") == identity.prompt_hashes(cond)       # unchanged for every non-cohort split
        assert identity.prompt_hashes(cond, split=o.SPLIT)["system_prompt" if name != "kaggle_zero_shot_legacy_prompt" else "zero_shot_prompt"] != \
            identity.prompt_hashes(cond)["system_prompt" if name != "kaggle_zero_shot_legacy_prompt" else "zero_shot_prompt"]


# ---------------------------------------------------------------------------------------------------------------- identity
def test_cohort_settings_are_required_on_the_cohort_and_refused_elsewhere():
    cond = C.get_condition("kaggle_zero_shot")
    with pytest.raises(ValueError, match="needs"):
        identity.run_settings(cond, "m", cohort="c", split=o.SPLIT)
    with pytest.raises(ValueError, match="must not"):
        identity.run_settings(cond, "m", cohort="c", split="test", cohort_settings=SETTINGS)
    st = settings("kaggle_zero_shot")
    assert st[K.SETTINGS_KEY] == SETTINGS and st["split"] == o.SPLIT and st["dataset"] == "dentex"
    assert K.SETTINGS_KEY not in identity.run_settings(cond, "m", cohort="c", split="test")


def test_a_new_coco_or_mapping_is_a_new_cohort_and_a_new_identity():
    other = {**SETTINGS, "mapping_sha256": "n" * 64}
    assert identity.cohort_sha([1, 2]) != identity.cohort_sha([1, 2], SETTINGS) != identity.cohort_sha([1, 2], other)
    assert identity.fingerprint(settings("kaggle_zero_shot")) != identity.fingerprint(settings("kaggle_zero_shot", cohort=other))
    with pytest.raises(identity.IdentityMismatch):
        identity.assert_same(settings("kaggle_zero_shot"), settings("kaggle_zero_shot", cohort=other))


def test_shard_specs_carry_the_cohort_settings_to_the_worker():
    specs = build_specs(condition="kaggle_zero_shot", image_ids=[1, 2], rel_paths={"1": "a", "2": "b"}, image_datasets=["o/d"],
                        cohort_sha256=identity.cohort_sha([1, 2], SETTINGS), split=o.SPLIT, cohort_settings=SETTINGS)
    again = type(specs[0]).from_json(specs[0].to_json())
    assert again.cohort_settings == SETTINGS and again.split == o.SPLIT
    plain = build_specs(condition="kaggle_zero_shot", image_ids=[1], rel_paths={"1": "a"}, image_datasets=["o/d"], cohort_sha256="c", split="test")
    assert plain[0].cohort_settings is None


# ------------------------------------------------------------------------------------------------------- what the model sees
@pytest.mark.parametrize("cond", ["kaggle_zero_shot", "kaggle_tools"])
def test_the_model_is_shown_the_cohort_system_prompt(table, tmp_path, cond):
    images, _, _ = table
    backend = FakeBackend([[FINAL]])
    coordinator.run_cell(backend=backend, condition=cond, images_df=images, image_ids=[100], out_dir=tmp_path / "c", settings=settings(cond), log=lambda s: None)
    assert "Root Canal" in backend.systems[0] and "Curettage" in backend.systems[0]
    base = FakeBackend([[FINAL]])
    coordinator.run_cell(backend=base, condition=cond, images_df=images, image_ids=[100], out_dir=tmp_path / "b",
                         settings=identity.run_settings(C.get_condition(cond), "fake/model-1", cohort="c", split="test", grounder_sha256="g" * 64), log=lambda s: None)
    assert "Root Canal" not in base.systems[0]


def test_the_legacy_zero_shot_user_prompt_is_the_cohort_one(table, tmp_path, monkeypatch):
    images, _, _ = table
    # only the reply parser of baselines.py is needed (its import pulls the training stack); the prompt is what is under test here
    monkeypatch.setitem(sys.modules, "dental_agent.evaluation.baselines", SimpleNamespace(parse_zero_shot_response=json.loads))
    backend = FakeBackend([[FINAL]])
    coordinator.run_cell(backend=backend, condition="kaggle_zero_shot_legacy_prompt", images_df=images, image_ids=[100], out_dir=tmp_path,
                         settings=settings("kaggle_zero_shot_legacy_prompt"), log=lambda s: None)
    sent = json.dumps([p for p in backend.sent[0][0]["parts"] if isinstance(p, str)] + [str(p) for p in backend.sent[0][0]["parts"]])
    assert "Root Canal" in sent


def test_preflight_renders_the_cohort_prompts():
    assert preflight.check_prompts_render(o.SPLIT)["ok"] and preflight.check_prompts_render("test")["ok"]


# --------------------------------------------------------------------------------------------------------------------- scoring
def test_cohort_scoring_names_the_class_and_adds_the_family_fields_and_other_splits_do_not(table, tmp_path):
    images, annots, cats = table
    ids = [100, 101, 102]
    outcome = {}
    for split, cohort in (("test", None), (o.SPLIT, SETTINGS)):
        st = identity.run_settings(C.get_condition("kaggle_zero_shot"), "fake/model-1", cohort=identity.cohort_sha(ids, cohort), split=split, cohort_settings=cohort)
        out = tmp_path / split
        coordinator.run_cell(backend=FakeBackend([[ROOT_CANAL], [EMPTY], [EMPTY]]), condition="kaggle_zero_shot", images_df=images, image_ids=ids, out_dir=out,
                             settings=st, log=lambda s: None)
        p = importer.cell_paths(out, "fake/model-1", "kaggle_zero_shot")
        recs = importer.import_transport(p["transport"], annots, cats, settings=st, split=split, cohort_ids=set(ids), evaluated_path=p["evaluated"])["records"]
        outcome[split] = {r["image_id"]: r for r in recs}
    plain, scored = outcome["test"][100], outcome[o.SPLIT][100]
    assert "scoring_profile" not in plain and not plain["exact_match"]               # the 4-class scorer cannot place "Root Canal Treatment"
    assert scored["exact_match"] and scored["scoring_profile"] == "test250_scoring_v2" and scored["n_renamed_predictions"] == 1
    assert scored["family_exact_f1"] == 1.0 and scored["split"] == o.SPLIT and scored["eval_settings"][K.SETTINGS_KEY] == SETTINGS
    assert scored["predictions"][0]["diagnosis"] != "Root Canal"                      # the record keeps the model's own words, as evaluate_condition does
    assert outcome[o.SPLIT][101]["all_exact_match"]                                   # a healthy image answered with no findings


def test_scored_records_match_what_score_cohort_and_the_shared_matcher_give(table):
    from dental_agent.evaluation.metrics import match_multi_findings
    from dental_agent.evaluation.test250_scoring import score_cohort
    images, annots, cats = table
    transport = records.transport_record(image_id=100, condition="kaggle_zero_shot", model_slug="m", traj=SimpleNamespace(
        final_answer=json.loads(ROOT_CANAL)["final_answer"], format_ok=True, turns=[], tool_calls=0, messages=[]), usage=[], seconds=0.0, identity={})
    rec = records.evaluated_record(transport, annots, cats, dataset="dentex", split=o.SPLIT, settings={K.SETTINGS_KEY: SETTINGS})
    pred = records.findings_from_trajectory(SimpleNamespace(final_answer=transport["final_answer"]))
    canon, extra = score_cohort(rec["ground_truth"], pred)
    m = match_multi_findings(rec["ground_truth"], canon)
    assert all(rec[k] == extra[k] for k in extra) and rec["exact_f1"] == m["exact_f1"] and rec["fdi_f1"] == m["fdi_f1"]


# ----------------------------------------------------------------------------------------------------------------- the loader
def test_load_split_uses_the_projects_cohort_resolver_and_loader(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    import tests.test_official_test250 as T
    files, _ = T.cohort_files(tmp_path)
    prepared = T.prep(tmp_path, files)
    calls = {}
    import dental_agent.harness.notebook_eval as ne
    monkeypatch.setattr(ne, "prepare_inputs", lambda data, split, dev, api, download, token, files=None: calls.update(split=split, data=data) or prepared)
    fake = SimpleNamespace(load_eval_split=lambda args: o.load_cohort_split(args) if args.split == o.SPLIT else ("plain", None, None))
    monkeypatch.setitem(sys.modules, "scripts.evaluate_models", fake)
    import scripts
    monkeypatch.setattr(scripts, "evaluate_models", fake, raising=False)
    images, annots, cats, cohort = K.load_split(o.SPLIT, data_dir=str(tmp_path), api=object(), download=object())
    assert calls["split"] == o.SPLIT and len(images) == 250 and cohort["scoring"] == "test250_scoring_v2" and cohort["coco_version"] == "v1"
    assert cohort["mapping_sha256"] == o.sha256_file(o.MAPPING_DIR / "official_test250_v1.json")
    other = K.load_split("test", data_dir=str(tmp_path))                              # any other split: no cohort resolver, no cohort settings
    assert other == ("plain", None, None, None) and calls["split"] == o.SPLIT
