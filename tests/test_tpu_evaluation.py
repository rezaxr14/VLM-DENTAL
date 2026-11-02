"""TPU evaluation wiring (patch 0063): offline checks of the gating, flags, identities, refusal records and the settings that keep
GPU/CPU result files and resumes exactly as they were. Nothing here touches XLA (see test_tpu_evaluation_xla_cpu.py)."""
import argparse
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from dental_agent.harness import notebook_eval as ne
from dental_agent.model import tpu_inference as ti
from dental_agent.model.local_runtime import LocalRuntime, add_runtime_args, runtime_from_args, runtime_record
from dental_agent.model.tpu_generation import GenerationTooLong, SlotOverflow, TPUGenerationConfig, install_tpu_generation
from scripts import evaluate_models as ev

TPU_DEVICE = [dict(name="TPU v5e-8", gib=0.0, capability=(0, 0), tpu=True)]
T4 = [dict(name="T4", gib=15.0, capability=(7, 5)), dict(name="T4", gib=15.0, capability=(7, 5))]


# ---- notebook gating, flags and identity -------------------------------------------------------------------------------

def test_tpu_profile_needs_a_tpu_and_never_runs_on_cuda():
    assert "TPU" in ne.hardware_reason("tpu", "sft", [])
    assert "TPU" in ne.hardware_reason("tpu", "sft", T4)             # a CUDA box does not satisfy the TPU profile
    assert ne.hardware_reason("tpu", "sft", TPU_DEVICE) is None
    assert ne.hardware_reason("tpu", "zeroshot", TPU_DEVICE) is None
    assert ne.hardware_reason("tpu", "rules_skills", TPU_DEVICE) is None
    assert "speculative" in ne.hardware_reason("tpu", "speculative", TPU_DEVICE)


def test_a_tpu_never_satisfies_a_cuda_profile_and_cuda_results_are_unchanged():
    assert "CUDA" in ne.hardware_reason("dual_t4", "sft", TPU_DEVICE)
    assert "CUDA" in ne.hardware_reason("workstation48_lora", "sft", TPU_DEVICE)
    assert ne.hardware_reason("dual_t4", "sft", T4) is None
    assert ne.hardware_reason("dual_t4", "sft", T4 + TPU_DEVICE) is None


def test_runtime_flags_carry_tpu_generation_only_for_the_tpu_profile():
    tpu = ne.runtime_flags("tpu")
    assert tpu[tpu.index("--tpu-generation-chunk-tokens") + 1] == "512"
    tiers = tpu[tpu.index("--tpu-generation-tiers") + 1:tpu.index("--tpu-vision-slots")]
    assert tiers == ["32768", "49152"]
    assert tpu[tpu.index("--tpu-vision-slots") + 1:] == ["8", "16", "6"]
    for gpu in ("dual_t4", "rtx4090_qlora", "workstation48_lora", "dual_t4_fp16"):
        assert not any(f.startswith("--tpu-") for f in ne.runtime_flags(gpu))


def test_tpu_flags_parse_into_the_runtime_and_gpu_records_keep_their_shape():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="bf16")                 # the evaluators define it themselves
    add_runtime_args(parser)
    args = parser.parse_args(ne.runtime_flags("tpu"))
    rt = runtime_from_args(args)
    assert rt.tpu_generation_chunk_tokens == 512 and rt.tpu_generation_tiers == (32768, 49152) and rt.tpu_vision_slots == (8, 16, 6)
    off = runtime_from_args(parser.parse_args([]))
    assert off.tpu_generation_chunk_tokens == 0
    assert LocalRuntime().tpu_generation_chunk_tokens == 0
    keys = ("tpu_generation_chunk_tokens", "tpu_generation_tiers", "tpu_vision_slots", "tpu_generation_warmup")
    gpu = runtime_record(SimpleNamespace(_dental_local_runtime=off))
    assert gpu and not any(k in gpu for k in keys)                      # GPU/CPU records keep their previous shape
    assert all(k in runtime_record(SimpleNamespace(_dental_local_runtime=rt)) for k in keys)


def test_grpo_flags_do_not_leak_into_the_evaluation_runtime():
    grpo = argparse.ArgumentParser()
    grpo.add_argument("--tpu-generation-chunk-tokens", type=int, default=512)
    grpo.add_argument("--tpu-generation-tiers", type=int, nargs="+", default=[8192, 16384, 32768])
    add_runtime_args(grpo, scoring=True, tpu_generation=False)
    rt = runtime_from_args(grpo.parse_args([]))
    assert rt.tpu_generation_chunk_tokens == 0 and rt.tpu_generation_tiers == (32768, 49152)


def test_resume_tolerates_settings_that_predate_the_tpu_flags_but_not_changed_ones():
    old = dict(max_new_tokens=16384, precision="qlora", seed=42, output="/old", adapter_path="/old", existing_results="resume")
    saved = dict(settings=old, image_ids=[1], conditions={})
    new = dict(old, output="/new", adapter_path="/new", canonical_letterbox=False, tpu_generation_chunk_tokens=0,
               tpu_generation_tiers=[32768, 49152], tpu_vision_slots=[8, 16, 6], tpu_generation_warmup=True)
    ne.validate_resume_settings(saved, new, [1])                      # GPU file written before the TPU flags existed
    with pytest.raises(ValueError, match="tpu_generation_chunk_tokens"):
        ne.validate_resume_settings(saved, dict(new, tpu_generation_chunk_tokens=512), [1])   # TPU run must not resume it
    with pytest.raises(ValueError, match="canonical_letterbox"):
        ne.validate_resume_settings(saved, dict(new, canonical_letterbox=True), [1])
    both = dict(old, canonical_letterbox=True, tpu_generation_chunk_tokens=512, tpu_generation_tiers=[32768], tpu_vision_slots=[8, 16, 6],
                tpu_generation_warmup=True)
    ne.validate_resume_settings(dict(settings=both, image_ids=[1], conditions={}), dict(both, output="/n"), [1])
    with pytest.raises(ValueError, match="tpu_generation_tiers"):
        ne.validate_resume_settings(dict(settings=both, image_ids=[1], conditions={}), dict(both, tpu_generation_tiers=[49152]), [1])


def test_original_size_is_unsupported_on_tpu_and_requires_canonical_letterbox_views():
    ti.require_static_views(canonical_resize=True, letterbox=True, what="x")
    for kw in (dict(canonical_resize=False, letterbox=True), dict(canonical_resize=True, letterbox=False)):
        with pytest.raises(ValueError, match="canonical"):
            ti.require_static_views(what="x", **kw)


# ---- refusals are recorded, counted and never silent ------------------------------------------------------------------

def test_refused_trajectory_scores_as_no_answer():
    traj = ti.refused_trajectory(7, "x")
    assert traj.image_id == 7 and traj.final_answer is None and traj.format_ok is False and traj.turns == []
    assert ev.findings_from_trajectory(traj) == []


def test_note_refusal_counts_and_names_the_image():
    model = SimpleNamespace()
    install_tpu_generation_stub(model)
    reason = ti.note_refusal(model, 11, GenerationTooLong("prompt 99999 + 16384 > 49152"))
    assert "GenerationTooLong" in reason
    stats = model._dental_tpu_generation.stats
    assert stats["rollouts_dropped"] == 1 and stats["refused_images"][0]["image_id"] == 11


def install_tpu_generation_stub(model):
    object.__setattr__(model, "_dental_tpu_generation",
                       SimpleNamespace(stats={"rollouts_dropped": 0}, chunk_tokens=64, cache_tiers=(256,), slot_budget={"FULL": 1}))


def test_generation_report_is_written_next_to_the_results(tmp_path):
    model = SimpleNamespace()
    install_tpu_generation_stub(model)
    ti.note_refusal(model, 3, SlotOverflow("too many images"))
    path = tmp_path / "sub" / "r.json"
    report = ti.write_generation_report(model, path)
    assert json.loads(path.read_text())["stats"]["rollouts_dropped"] == 1 and report["chunk_tokens"] == 64
    assert ti.write_generation_report(SimpleNamespace(), tmp_path / "none.json") == {} and not (tmp_path / "none.json").exists()


@pytest.fixture
def world(tmp_path, monkeypatch):
    rows = []
    for i in (1, 2, 3):
        f = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(f)
        rows.append({"id": i, "local_path": str(f)})
    gt = {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}
    monkeypatch.setattr(ev, "extract_ground_truth", lambda image_id, a, c: [dict(gt)])

    def fake(model, processor, images_df, image_id, registry, args):
        if image_id == 3:
            raise GenerationTooLong("prompt 60000 + 16384 > 49152")
        return SimpleNamespace(final_answer=[dict(gt, confidence=0.9)], format_ok=True, tool_calls=0,
                               turns=[{"turn": 0, "raw_output": "{}", "status": "final_answer", "tool_calls_this_turn": []}],
                               messages=[])

    monkeypatch.setattr(ev, "run_condition_trajectory", fake)
    args = argparse.Namespace(
        dataset="dentex", split="test", output_dir=str(tmp_path / "out"), reuse_cached=False, limit=None, model_id="m",
        adapter_path=None, precision="bf16", canonical_resize=True, canonical_letterbox=True, max_tool_calls=50, max_turns=None,
        max_new_tokens=16384, malformed_retries=0, existing_results="error", condition="base_no_tools",
        tpu_generation_chunk_tokens=512, tpu_generation_tiers=[32768, 49152], tpu_vision_slots=[8, 16, 6])
    return SimpleNamespace(args=args, imgs=pd.DataFrame(rows), out=tmp_path / "out")


def _run(w, cond="base_no_tools"):
    return ev.evaluate_condition(cond, w.args, w.imgs, pd.DataFrame(), pd.DataFrame(), loaded=(SimpleNamespace(), object()))


def test_eval_records_a_refused_episode_flags_it_and_keeps_going(world):
    records = _run(world)
    assert [r["image_id"] for r in records] == [1, 2, 3]
    refused = records[2]
    assert "GenerationTooLong" in refused["generation_refused"]
    assert refused["format_ok"] is False and refused["predictions"] == [] and refused["exact_match"] is False
    assert all("generation_refused" not in r for r in records[:2])
    assert records[0]["exact_match"] is True
    assert records[0]["eval_settings"]["tpu_generation"] == {"chunk_tokens": 512, "cache_tiers": [32768, 49152], "vision_slots": [8, 16, 6]}
    saved = [json.loads(line) for line in next(world.out.glob("eval_*_base_no_tools_canonical.jsonl")).read_text().splitlines()]
    assert saved[2]["generation_refused"] == refused["generation_refused"]


def test_eval_on_tpu_needs_canonical_letterbox_views_before_loading_anything(world):
    world.args.canonical_letterbox = False
    with pytest.raises(ValueError, match="canonical"):
        _run(world)
    world.args.canonical_letterbox, world.args.canonical_resize = True, False
    with pytest.raises(ValueError, match="canonical"):
        _run(world)


def test_gpu_settings_and_files_are_unchanged_without_tpu_generation(world):
    args = world.args
    args.tpu_generation_chunk_tokens = 0
    assert "tpu_generation" not in ev.eval_settings(args, "base_no_tools", None)
    delattr(args, "tpu_generation_chunk_tokens")                       # a caller that never heard of the flag
    assert "tpu_generation" not in ev.eval_settings(args, "base_no_tools", None)


def test_tpu_evaluation_without_the_generator_flag_stops_before_loading_weights():
    with pytest.raises(SystemExit, match="tpu-generation-chunk-tokens"):
        ev.load_eval_model("m", None, SimpleNamespace(type="xla"), None, "base_no_tools", runtime_options=LocalRuntime())


# ---- rules/skills runner ------------------------------------------------------------------------------------------------

def test_harness_episode_uses_the_requested_letterbox_and_records_a_refusal(monkeypatch):
    import dental_agent.agent.loop as loop
    from dental_agent.harness import evaluation as he
    from dental_agent.utils import canonical

    monkeypatch.setenv(canonical.CANONICAL_LETTERBOX_ENV, "1")        # restored on teardown
    seen = []

    def fake(image_id, images, model, processor, **kw):
        seen.append(canonical.letterbox_enabled())
        raise SlotOverflow("9 FULL views > 8 slots")

    monkeypatch.setattr(loop, "run_agent", fake)
    monkeypatch.setattr("dental_agent.evaluation.ground_truth.extract_ground_truth", lambda *a: [])
    model = SimpleNamespace()
    install_tpu_generation_stub(model)
    record, traj = he.evaluate_episode(5, None, None, None, model=model, processor=None, letterbox=False)
    assert seen == [False]                                              # the GPU default is kept ...
    he.evaluate_episode(5, None, None, None, model=model, processor=None, letterbox=True)
    assert seen == [False, True]                                        # ... and the TPU profile asks for bars
    assert "SlotOverflow" in record["generation_refused"] and record["format_ok"] is False and record["predictions"] == []
    assert model._dental_tpu_generation.stats["rollouts_dropped"] == 2
    summary = he.pooled([record, dict(record, image_id=6, generation_refused="")])
    assert summary["generation_refused"] == 1 and summary["episodes"] == 2
    clean = dict(record); clean.pop("generation_refused")
    assert "generation_refused" not in he.pooled([clean])               # GPU summaries keep their exact keys


def test_run_dev_exposes_the_letterbox_flag_with_the_gpu_default():
    from scripts.harness import run_dev
    base = ["--model-id", "m", "--adapter-path", "a"]
    assert run_dev.parse_args(base).canonical_letterbox is False
    assert run_dev.parse_args(base + ["--canonical-letterbox"]).canonical_letterbox is True
    args = run_dev.parse_args(base + ne.runtime_flags("tpu") + ["--canonical-letterbox"])
    assert args.tpu_generation_chunk_tokens == 512


def test_zero_shot_parser_accepts_the_tpu_flags_and_defaults_to_the_original_view():
    from scripts import run_zero_shot
    parser = run_zero_shot.build_parser()
    args = parser.parse_args(["--provider", "transformers"])
    assert args.canonical_resize is False and args.tpu_generation_chunk_tokens == 0
    flags = ne.runtime_flags("tpu"); flags[0] = "--local-precision"
    tpu = parser.parse_args(["--provider", "transformers", *flags, "--canonical-resize", "--canonical-letterbox"])
    assert tpu.canonical_resize and tpu.canonical_letterbox and tpu.tpu_generation_chunk_tokens == 512
