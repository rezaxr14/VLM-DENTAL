"""GPU image view (patch 0036): on a CUDA GPU the model is never shown black bars, and nothing is stretched.

* ``--no-canonical-letterbox``: canonical size = uniform scale to fit the canvas, NOTHING added (no bars), each side a multiple
  of 32 so the Qwen processor leaves it alone (one resample). Aspect ratio kept up to that rounding (<= 16 px per side).
* ``--no-canonical-resize``: the full image, handed to the processor untouched (the existing native mode).
* The padded canvas (default, TPU static slots) is unchanged, and the training scripts / doctor refuse it on CUDA.
"""

import json
from argparse import Namespace

import pytest
from PIL import Image

from dental_agent.utils import canonical as C
from dental_agent.utils import inputs as inp


@pytest.fixture(autouse=True)
def _reset_letterbox(monkeypatch):
    monkeypatch.delenv(C.CANONICAL_LETTERBOX_ENV, raising=False)
    yield


SIZES = [((2872, 1504), "FULL"), ((2900, 1250), "FULL"), ((1536, 768), "FULL"), ((180, 200), "CROP"), ((300, 260), "CROP"),
         ((400, 200), "COMPARE"), ((600, 700), "COMPARE"), ((5000, 1500), "FULL")]


@pytest.mark.parametrize("size,family", SIZES)
def test_no_bars_view_is_the_scaled_image_alone(size, family):
    cw, ch = C.CANONICAL_SIZES[family]
    white = Image.new("RGB", size, (255, 255, 255))
    out = C.to_canonical(white, family, letterbox=False)
    assert out.width <= cw and out.height <= ch
    assert out.width % 32 == 0 and out.height % 32 == 0                       # the processor's own grid: it will not resample again
    assert out.getextrema() == ((255, 255), (255, 255), (255, 255))             # not one black pixel was added
    scale = min(cw / size[0], ch / size[1])
    assert abs(out.width - size[0] * scale) <= 16 and abs(out.height - size[1] * scale) <= 16   # aspect kept up to the 32-px rounding


@pytest.mark.parametrize("size,family", SIZES)
def test_padded_canvas_is_unchanged_and_still_has_bars(size, family):
    out = C.to_canonical(Image.new("RGB", size, (255, 255, 255)), family, letterbox=True)
    assert out.size == C.CANONICAL_SIZES[family]
    if size != C.CANONICAL_SIZES[family] and abs(size[0] / size[1] - C.CANONICAL_SIZES[family][0] / C.CANONICAL_SIZES[family][1]) > 0.02:
        assert out.getextrema()[0][0] == 0                                      # black bars exist in this mode (TPU static slots)


def test_worked_example_full_panorama():
    """2872 x 1504 -> scale min(1536/2872, 768/1504) = 0.5106 -> 1466.5 x 768 -> 1472 x 768: 46 x 24 patches = 1104 tokens (the padded
    canvas is 1152). 1.9096 -> 1.9167 aspect (0.4 %)."""
    out = C.to_canonical(Image.new("RGB", (2872, 1504)), "FULL", letterbox=False)
    assert out.size == (1472, 768) and (out.width // 32) * (out.height // 32) == 1104


def test_an_image_already_on_the_canvas_is_returned_as_it_is():
    img = Image.new("RGB", (1536, 768), (7, 7, 7))
    assert C.to_canonical(img, "FULL", letterbox=False) is img and C.to_canonical(img, "FULL", letterbox=True) is img


def test_setting_defaults_to_padded_and_follows_set_letterbox():
    assert C.letterbox_enabled() is True                                         # unchanged behaviour for every existing command
    C.set_letterbox(False)
    assert C.letterbox_enabled() is False
    assert C.to_canonical(Image.new("RGB", (2872, 1504)), "FULL").size == (1472, 768)   # letterbox=None reads the run setting
    C.set_letterbox(True)
    assert C.to_canonical(Image.new("RGB", (2872, 1504)), "FULL").size == (1536, 768)


# ---------------------------------------------------------------------------------------------------------------------
# GPU refusal
# ---------------------------------------------------------------------------------------------------------------------
def test_gpu_policy_refuses_black_bars_on_cuda_and_names_the_fix():
    with pytest.raises(SystemExit, match="--no-canonical-letterbox"):
        C.check_gpu_image_policy(True, True, cuda=True, tpu=False, what="SFT")


@pytest.mark.parametrize("canonical,letterbox,cuda,tpu", [
    (True, False, True, False),    # GPU, canonical size, nothing added
    (False, True, True, False),    # GPU, full image (letterbox is irrelevant without canonical views)
    (False, False, True, False),
    (True, True, False, True),     # TPU needs the bars
    (True, True, False, False),    # CPU smoke run
])
def test_gpu_policy_allows_everything_else(canonical, letterbox, cuda, tpu):
    C.check_gpu_image_policy(canonical, letterbox, cuda=cuda, tpu=tpu)


def test_train_sft_and_grpo_refuse_bars_on_cuda(monkeypatch):
    """The scripts call the policy before anything heavy loads (source-level guard, since CUDA is not available here)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for f, needle in [("scripts/train_sft.py", "check_gpu_image_policy(canonical_resize, args.canonical_letterbox"),
                      ("scripts/run_grpo.py", "check_gpu_image_policy(args.canonical_resize, args.canonical_letterbox"),
                      ("scripts/run_grpo_sweep.py", "check_gpu_image_policy(args.canonical_resize, args.canonical_letterbox"),
                      ("dental_agent/training/grpo.py", "check_gpu_image_policy(canonical_resize, letterbox_enabled()")]:
        assert needle in (root / f).read_text(), f


def test_cli_flag_default_is_padded_and_can_be_switched_off():
    import argparse
    ap = argparse.ArgumentParser()
    C.add_letterbox_arg(ap)
    assert ap.parse_args([]).canonical_letterbox is True
    assert ap.parse_args(["--no-canonical-letterbox"]).canonical_letterbox is False
    assert ap.parse_args(["--canonical-letterbox"]).canonical_letterbox is True


# ---------------------------------------------------------------------------------------------------------------------
# Dataset: what the processor is actually handed
# ---------------------------------------------------------------------------------------------------------------------
def _dataset(tmp_path, size):
    import test_canonical_pipeline as tcp
    from dental_agent.training.sft import DentalSFTDataset

    native = Image.new("RGB", size, (0, 0, 0))
    native.paste((255, 255, 255), (300, 200, 400, 350))
    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps(tcp._record(tmp_path, native)) + "\n")
    proc = tcp._RecordingProcessor()
    return DentalSFTDataset(trace, processor=proc, data_dir=tmp_path, canonical_resize=True), proc


def test_dataset_without_bars_hands_the_processor_unpadded_views_and_tools_still_see_native_pixels(tmp_path):
    C.set_letterbox(False)
    ds, proc = _dataset(tmp_path, (1000, 600))
    ds[0]
    base, window, crop = proc.seen_images
    assert base.size == (1280, 768) and window.size == (1280, 768)             # 1000x600 * 1.28; the padded view would be 1536x768
    assert base.getextrema() == ((0, 255), (0, 255), (0, 255)) and base.getpixel((0, 0)) == (0, 0, 0)
    assert min(base.getpixel((x, y))[0] for x in (0, base.width - 1) for y in (0, base.height - 1)) == 0
    assert crop.width % 32 == 0 and crop.height % 32 == 0 and crop.width <= 256 and crop.height <= 384
    assert crop.getpixel((crop.width // 2, crop.height // 2)) == (255, 255, 255)   # the tool ran on the native image, not on a resized one


def test_dataset_with_bars_is_unchanged(tmp_path):
    ds, proc = _dataset(tmp_path, (1000, 600))
    ds[0]
    assert [im.size for im in proc.seen_images] == [(1536, 768), (1536, 768), (256, 384)]


def test_dataset_ignores_a_manifest_measured_with_the_other_letterbox_setting(tmp_path):
    from dental_agent.training.sft import DentalSFTDataset
    import test_canonical_pipeline as tcp

    ds, _ = _dataset(tmp_path, (1000, 600))
    trace = tmp_path / "t.jsonl"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"canonical_resize": True, "canonical_letterbox": True, "lengths_by_image_id": {"7": 99999}}))
    C.set_letterbox(False)
    with pytest.warns(UserWarning, match="canonical_letterbox"):
        ds = DentalSFTDataset(trace, processor=tcp._RecordingProcessor(), data_dir=tmp_path, max_seq_len=1000,
                              token_lengths_manifest=manifest, canonical_resize=True)
    assert len(ds) == 1                                                          # the wrong lengths filtered nothing
    manifest.write_text(json.dumps({"canonical_resize": True, "lengths_by_image_id": {"7": 99999}}))   # legacy = measured with bars
    with pytest.warns(UserWarning, match="Ignoring token-length manifest"):
        DentalSFTDataset(trace, processor=tcp._RecordingProcessor(), data_dir=tmp_path, max_seq_len=1000,
                         token_lengths_manifest=manifest, canonical_resize=True)


# ---------------------------------------------------------------------------------------------------------------------
# Manifest staleness, doctor
# ---------------------------------------------------------------------------------------------------------------------
def _fresh_manifest(trace, letterbox):
    fp = inp.manifest_fingerprint([trace], True, letterbox)
    return {"canonical_resize": True, "canonical_letterbox": letterbox, "_meta": {"fingerprint": fp, "complete": True}}


def test_manifest_staleness_notices_the_letterbox_setting(tmp_path):
    trace = tmp_path / "a.jsonl"
    trace.write_text('{"x": 1}\n')
    m = _fresh_manifest(trace, True)
    assert inp.manifest_staleness(m, [trace], True, canonical_letterbox=True) == []
    reasons = inp.manifest_staleness(m, [trace], True, canonical_letterbox=False)
    assert any("canonical_letterbox=True" in r for r in reasons)
    legacy = {k: v for k, v in m.items() if k != "canonical_letterbox"}            # written before the switch: measured with bars
    assert inp.manifest_staleness(legacy, [trace], True, canonical_letterbox=True) == []
    assert inp.manifest_staleness(legacy, [trace], True, canonical_letterbox=False)
    native = {**m, "canonical_resize": False}
    assert not any("letterbox" in r for r in inp.manifest_staleness(native, [trace], False, canonical_letterbox=False))


def test_manifest_command_carries_the_flag():
    assert inp.manifest_command("data", True, "M", [5, 10, 4]).endswith("--canonical-resize")
    assert inp.manifest_command("data", True, "M", [5, 10, 4], False).endswith("--canonical-resize --no-canonical-letterbox")
    assert "letterbox" not in inp.manifest_command("data", False, "M", [5, 10, 4], False)


def _rows(rep):
    return {name: (level, detail) for level, name, detail in rep.rows}


def test_doctor_fails_bars_on_cuda_passes_everything_else(monkeypatch):
    import torch
    from dental_agent.training import preflight as pf

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def view(**kw):
        rep = pf._Report(lambda line: None)
        args = Namespace(canonical_resize=True, canonical_letterbox=True, pad_vision_to_slots=False, **kw)
        ok = pf._check_image_view(args, rep)
        return ok, _rows(rep)
    ok, rows = view()
    assert not ok and rows["GPU image view"][0] == "FAIL" and "--no-canonical-letterbox" in rows["GPU image view"][1]
    rep = pf._Report(lambda line: None)
    ok = pf._check_image_view(Namespace(canonical_resize=True, canonical_letterbox=False, pad_vision_to_slots=False), rep)
    assert ok and _rows(rep)["image view"][0] == "PASS"
    rep = pf._Report(lambda line: None)
    assert pf._check_image_view(Namespace(canonical_resize=False, canonical_letterbox=True, pad_vision_to_slots=False), rep)
    rep = pf._Report(lambda line: None)
    assert not pf._check_image_view(Namespace(canonical_resize=True, canonical_letterbox=False, pad_vision_to_slots=True), rep)
    assert _rows(rep)["pad-vision-to-slots"][0] == "FAIL"


def test_grpo_doctor_compares_letterbox_with_the_sft_run(tmp_path):
    from dental_agent.training import preflight as pf

    def rows(meta, grpo_letterbox):
        (tmp_path / "sft_run_meta.json").write_text(json.dumps({"track": "with_tools", "canonical_resize": True, **meta}))
        rep = pf._Report(lambda line: None)
        args = Namespace(track="with_tools", canonical_resize=True, canonical_letterbox=grpo_letterbox, dev_split=None,
                         max_seq_len=16384, pad_vision_to_slots=False, vision_slots=(5, 10, 4), precision="bf16")
        pf._check_sft_consistency(args, tmp_path, set(), rep)
        return _rows(rep)
    assert rows({"canonical_letterbox": False}, False)["canonical-letterbox vs SFT"][0] == "PASS"
    assert rows({"canonical_letterbox": False}, True)["canonical-letterbox vs SFT"][0] == "FAIL"
    assert rows({}, True)["canonical-letterbox vs SFT"][0] == "PASS"                  # SFT written before the switch = bars
    assert rows({}, False)["canonical-letterbox vs SFT"][0] == "FAIL"
