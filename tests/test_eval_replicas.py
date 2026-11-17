"""Data-parallel evaluation helpers: the 4B model runs one single-GPU replica per GPU (no PCIe traffic); 9B is unchanged."""
import json
import sys
from pathlib import Path

import pytest

from dental_agent.harness import eval_replicas as er
from dental_agent.harness import notebook_eval as ne


def _rec(i, **kw):
    return (json.dumps({"image_id": i, **kw}) + "\n").encode()


def test_replicas_only_for_4b_and_only_with_the_profile_key():
    cfg = {"replicas": 2}
    gpus = [{"gib": 15}, {"gib": 15}]
    assert er.eval_replicas(cfg, gpus, "9b") == 1          # 9B does not fit one T4: layer split as before
    assert er.eval_replicas(cfg, gpus, "4b") == 2
    assert er.eval_replicas({}, gpus, "4b") == 1
    assert er.eval_replicas(cfg, [{"gib": 15}], "4b") == 1  # never more replicas than GPUs
    assert er.eval_replicas(cfg, None, "4b") == 2


def test_profile_tables_declare_replicas_on_dual_t4_only():
    for name in ("dual_t4", "dual_t4_fp16", "dual_t4_ddp_fp16_4b"):
        assert ne.PROFILES[name]["replicas"] == 2
    for name, cfg in ne.PROFILES.items():
        if not name.startswith("dual_t4"):
            assert cfg.get("replicas", 1) == 1


def test_notebook_tables_match_module_replicas():
    import glob
    for path in glob.glob("notebooks/**/*.ipynb", recursive=True):
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            src = "".join(cell["source"]) if cell["cell_type"] == "code" else ""
            if '"dual_t4_ddp_fp16_4b": {' in src and '"tag"' in src and "EVAL_PROFILES" in src:
                for key in ("dual_t4", "dual_t4_fp16", "dual_t4_ddp_fp16_4b"):
                    block = src.split(f'"{key}": {{', 1)[1].split("}", 1)[0]
                    assert '"replicas": 2' in block, (path, key)


def test_replica_commands_pin_one_gpu_each_and_split_images(tmp_path):
    cmd = [sys.executable, "scripts/evaluate_models.py", "--output-dir", "out", "--x", "1"]
    jobs = er.replica_commands(cmd, "sft", tmp_path, tmp_path / "res.jsonl", 2)
    assert [j["env"]["CUDA_VISIBLE_DEVICES"] for j in jobs] == ["0", "1"]
    for i, j in enumerate(jobs):
        assert j["command"][-4:] == ["--shard-index", str(i), "--shard-count", "2"]
        assert j["command"][j["command"].index("--output-dir") + 1] == str(tmp_path / "_replicas" / f"gpu{i}")
    with pytest.raises(ValueError):
        er.replica_commands(cmd, "speculative", tmp_path, tmp_path / "r", 2)


def test_zeroshot_replicas_use_slices(tmp_path):
    cmd = ["python", "scripts/run_zero_shot.py", "--output-dir", "o", "--output", "r.jsonl"]
    jobs = er.replica_commands(cmd, "zeroshot", tmp_path, tmp_path / "r.jsonl", 2)
    assert jobs[1]["command"][-4:] == ["--total-slices", "2", "--slice-index", "2"]
    assert jobs[0]["result"].name == "results.jsonl"


def test_merge_is_ordered_unions_and_drops_torn_lines(tmp_path):
    result = tmp_path / "res.jsonl"
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_bytes(_rec(1) + _rec(3) + b'{"image_id": 5')    # torn final line
    b.write_bytes(_rec(2) + _rec(4))
    assert er.merge_replicas(result, [a, b], [1, 2, 3, 4, 5]) == 4
    assert [json.loads(l)["image_id"] for l in result.read_text().splitlines()] == [1, 2, 3, 4]


def test_merge_with_nothing_creates_no_file_and_rejects_foreign_images(tmp_path):
    result = tmp_path / "res.jsonl"
    assert er.merge_replicas(result, [tmp_path / "none.jsonl"], [1]) == 0
    assert not result.exists()
    s = tmp_path / "s.jsonl"
    s.write_bytes(_rec(9))
    with pytest.raises(ValueError):
        er.merge_replicas(result, [s], [1, 2])


def test_seed_replicas_hands_finished_records_to_every_replica(tmp_path):
    result = tmp_path / "res.jsonl"
    result.write_bytes(_rec(1) + _rec(2))
    jobs = er.replica_commands(["p", "--output-dir", "o"], "baseline", tmp_path, result, 2)
    er.seed_replicas(result, jobs)
    for j in jobs:
        assert j["result"].read_bytes() == _rec(1) + _rec(2)
    er.seed_replicas(result, jobs)                            # idempotent
    assert jobs[0]["result"].read_bytes() == _rec(1) + _rec(2)


def test_result_path_for_4b_is_a_separate_folder_and_9b_is_unchanged(monkeypatch):
    from dental_agent.utils import model_size
    ident = {"split": "test", "profile": "dual_t4_fp16", "runtime": {"tag": "dual_t4_fp16"}, "stage": "base", "kind": "zeroshot",
             "condition": "c", "size": "canonical"}
    monkeypatch.setenv("DENTAL_MODEL_SIZE", "9b")
    p9 = ne.result_path(ident)
    assert "/dual_t4_fp16/base/" in p9
    monkeypatch.setenv("DENTAL_MODEL_SIZE", "4b")
    p4 = ne.result_path(ident)
    assert "/dual_t4_fp16_qwen3_5_4b/base/" in p4
    assert p4.split("/")[:3] == p9.split("/")[:3]
