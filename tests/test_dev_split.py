"""Dev split: deterministic, test-set-safe, keyed by (dataset, image_id)."""

import json

import pandas as pd
import pytest

from dental_agent.data.dev_split import (
    assert_not_test_split,
    filter_records,
    image_key,
    load_dev_keys,
    make_dev_split,
    record_key,
    save_dev_split,
    select_training_pool,
)


def _train_frames(n_images=100, n_healthy=10, source="quadrant_enumeration_disease/train_disease.json"):
    images = pd.DataFrame({
        "id": list(range(1, n_images + 1)),
        "file_name": [f"train_{i}.png" for i in range(1, n_images + 1)],
        "local_path": [f"/x/train_{i}.png" for i in range(1, n_images + 1)],
        "source_file": source,
    })
    path_ids = list(range(n_healthy + 1, n_images + 1))  # ids 1..n_healthy are healthy
    annots = pd.DataFrame({"image_id": path_ids, "bbox": [[0, 0, 1, 1]] * len(path_ids)})
    return images, annots


def test_key_format_and_normalisation():
    assert image_key("DENTEX", 7) == "dentex::7"
    assert image_key("dentex", 7.0) == "dentex::7"
    assert record_key({"image_id": 12}) == "dentex::12"
    assert record_key({"image_id": 12, "dataset": "tufts"}) == "tufts::12"


def test_split_is_deterministic_and_stratified():
    images, annots = _train_frames(100, 10)
    a = make_dev_split(images, annots, n=20, seed=42)
    b = make_dev_split(images, annots, n=20, seed=42)
    c = make_dev_split(images, annots, n=20, seed=43)
    assert a == b
    assert a["image_ids"] != c["image_ids"]
    assert a["n"] == 20
    # 10 healthy of 100 -> round(20 * 0.10) = 2 healthy + 18 pathological
    assert (a["n_healthy"], a["n_pathological"]) == (2, 18)
    assert a["keys"] == [f"dentex::{i}" for i in a["image_ids"]]
    healthy_ids = set(range(1, 11))
    assert len(set(a["image_ids"]) & healthy_ids) == 2


def test_roundtrip_and_missing_file(tmp_path):
    images, annots = _train_frames()
    split = make_dev_split(images, annots, n=20)
    p = save_dev_split(split, tmp_path / "splits" / "dev.json")
    assert load_dev_keys(p) == set(split["keys"])
    assert load_dev_keys(None) == set()
    with pytest.raises(FileNotFoundError):
        load_dev_keys(tmp_path / "nope.json")


def test_filter_records_uses_dataset_and_id():
    keys = {"dentex::5"}
    recs = [{"image_id": 5, "dataset": "dentex"}, {"image_id": 5, "dataset": "tufts"}, {"image_id": 6}]
    kept, dropped = filter_records(recs, keys)
    assert dropped == 1
    assert [record_key(r) for r in kept] == ["tufts::5", "dentex::6"]


def test_guard_refuses_validation_file_but_not_train_file():
    train, _ = _train_frames()
    assert_not_test_split(train)
    test_like, _ = _train_frames(source="data/dentex/validation_triple.json")
    with pytest.raises(SystemExit):
        assert_not_test_split(test_like)
    with pytest.raises(SystemExit):
        make_dev_split(*[test_like, _train_frames()[1]], n=5)
    for name in ("train.json", "train_quadrant_enumeration_disease.json", "quadrant-enumeration-disease/train_3.json"):
        ok, _ = _train_frames(source=name)
        assert_not_test_split(ok)


def test_training_pool_excludes_dev_and_is_seeded():
    images, annots = _train_frames(100, 10)
    split = make_dev_split(images, annots, n=20)
    dev = set(split["image_ids"])
    pool = select_training_pool(images, annots, dev_keys=set(split["keys"]))
    assert len(pool) == 80
    assert not (set(pool["id"]) & dev)
    # healthy kept by default, droppable
    assert pool["id"].isin(range(1, 11)).any()
    no_h = select_training_pool(images, annots, dev_keys=set(split["keys"]), include_healthy=False)
    assert not no_h["id"].isin(range(1, 11)).any()
    # --max-images is a seeded random subset, reproducible, not "the first N ids"
    a = select_training_pool(images, annots, max_images=15, seed=7)
    b = select_training_pool(images, annots, max_images=15, seed=7)
    c = select_training_pool(images, annots, max_images=15, seed=8)
    assert list(a["id"]) == list(b["id"]) and len(a) == 15
    assert list(a["id"]) != list(c["id"])
    assert list(a["id"]) != list(range(1, 16))


def test_training_pool_refuses_test_split():
    test_like, annots = _train_frames(source="validation_triple.json")
    with pytest.raises(SystemExit):
        select_training_pool(test_like, annots)


def test_sft_dataset_excludes_dev_traces(tmp_path):
    sft = pytest.importorskip("dental_agent.training.sft")
    traces = tmp_path / "t.jsonl"
    rows = [{"image_id": i, "dataset": "dentex", "messages": []} for i in range(1, 6)]
    rows.append({"image_id": 3, "dataset": "tufts", "messages": []})
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = sft.DentalSFTDataset(traces, processor=None, data_dir=str(tmp_path), exclude_keys={"dentex::3", "dentex::5"})
    assert ds.dev_excluded == 2
    assert sorted((r["dataset"], r["image_id"]) for r in ds.records) == [
        ("dentex", 1), ("dentex", 2), ("dentex", 4), ("tufts", 3),
    ]
