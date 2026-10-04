"""Canonical resizing, TPU static vision-slot padding, vision-info extraction and XLA solve shim."""

import json
import warnings

import pytest
import torch
from PIL import Image

from dental_agent.model.backbone import safe_process_vision_info
from dental_agent.training import xla_patches as xp
from dental_agent.training.sft import BucketedQwenVLCollator, DentalSFTDataset
from dental_agent.utils import canonical as C

IMG_TOK = 99999


# --------------------------------------------------------------------------- canonical.py
def test_canonical_geometry_matches_project_spec():
    assert C.patches_per_image("FULL") == 4608 and C.tokens_per_image("FULL") == 1152
    assert C.patches_per_image("CROP") == 384 and C.tokens_per_image("CROP") == 96
    assert C.patches_per_image("COMPARE") == 768 and C.tokens_per_image("COMPARE") == 192
    assert (C.TOTAL_SLOTS, C.TOTAL_PATCHES, C.TOTAL_VISION_TOKENS) == (19, 29952, 7488)
    assert C.family_from_grid((1, 48, 96)) == "FULL"
    assert C.family_from_grid((1, 24, 16)) == "CROP"
    assert C.family_from_grid((1, 24, 32)) == "COMPARE"
    assert C.family_from_grid((1, 10, 10)) is None
    assert C.to_canonical(Image.new("RGB", (2872, 1504)), "FULL").size == (1536, 768)


# --------------------------------------------------------------------------- vision info
def test_safe_process_vision_info_passes_pil_images_through_unresized():
    img = Image.new("RGB", (1536, 768))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
    images, videos = safe_process_vision_info(msgs)
    assert videos is None and len(images) == 1 and images[0].size == (1536, 768)


def test_safe_process_vision_info_no_images_is_none_and_non_pil_failure_warns():
    assert safe_process_vision_info([{"role": "user", "content": "hi"}]) == (None, None)
    msgs = [{"role": "user", "content": [{"type": "image", "image": "/nonexistent.png"}]}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = safe_process_vision_info(msgs)
    assert out == (None, None) or out[0] is not None  # qwen_vl_utils may or may not be installed
    if out == (None, None):
        assert any(issubclass(x.category, RuntimeWarning) for x in w)


# --------------------------------------------------------------------------- dataset
class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [10] * max(len(text) // 4, 1)

    def decode(self, token_ids):
        return "dummy text"


class _RecordingProcessor:
    image_token_id = IMG_TOK

    def __init__(self):
        self.tokenizer = _Tok()
        self.seen_images = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "x" * 80

    def __call__(self, text, images=None, videos=None, padding=False, return_tensors=None, **kw):
        self.seen_images = list(images or [])
        ids = torch.full((1, 20), 10, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _record(tmp_path, native):
    p = tmp_path / "s.png"
    native.save(p)
    asst1 = {"thought": "t", "tool_calls": [
        {"tool": "window_level", "args": {"preset": "bone"}},
        {"tool": "zoom_crop", "args": {"bbox": [300.0, 200.0, 100.0, 150.0], "padding_frac": 0.2}},
    ]}
    return {
        "image_id": 7, "image_path": str(p),
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "image", "image": "<Image>"}, {"type": "text", "text": "Analyze."}]},
            {"role": "assistant", "content": json.dumps(asst1)},
            {"role": "user", "content": [
                {"type": "image", "image": "<Image>"}, {"type": "text", "text": "Result of window_level:"},
                {"type": "image", "image": "<Image>"}, {"type": "text", "text": "Result of zoom_crop:"}]},
            {"role": "assistant", "content": json.dumps({"thought": "d", "final_answer": []})},
        ],
        "turns": [], "final_answer": [],
    }


def _build(tmp_path, canonical):
    native = Image.new("RGB", (1200, 600), (0, 0, 0))
    native.paste((255, 255, 255), (300, 200, 400, 350))  # exactly the bbox region, native coordinates
    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps(_record(tmp_path, native)) + "\n")
    proc = _RecordingProcessor()
    ds = DentalSFTDataset(trace, processor=proc, data_dir=tmp_path, canonical_resize=canonical)
    return ds, proc


def test_dataset_native_by_default(tmp_path):
    ds, proc = _build(tmp_path, canonical=False)
    ds[0]
    assert proc.seen_images[0].size == (1200, 600)


def test_dataset_canonical_views_but_tools_run_on_native_image(tmp_path):
    ds, proc = _build(tmp_path, canonical=True)
    ds[0]
    sizes = [im.size for im in proc.seen_images]
    assert sizes == [(1536, 768), (1536, 768), (256, 384)]  # base FULL, window_level FULL, zoom_crop CROP
    crop = proc.seen_images[2]
    # If the tool had run on an already-resized image, bbox (native coords) would miss the white box.
    assert crop.getpixel((128, 192)) == (255, 255, 255)


def test_dataset_ignores_manifest_with_mismatched_resolution_mode(tmp_path):
    ds_native, _ = _build(tmp_path, canonical=False)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"canonical_resize": False, "lengths_by_image_id": {"7": 99999}}))
    trace = tmp_path / "t.jsonl"
    with pytest.warns(UserWarning, match="Ignoring token-length manifest"):
        ds = DentalSFTDataset(trace, processor=_RecordingProcessor(), data_dir=tmp_path,
                              max_seq_len=1000, token_lengths_manifest=manifest, canonical_resize=True)
    assert len(ds) == 1  # not filtered by stale native-resolution length
    ds2 = DentalSFTDataset(trace, processor=_RecordingProcessor(), data_dir=tmp_path,
                           max_seq_len=1000, token_lengths_manifest=manifest, canonical_resize=False)
    assert len(ds2) == 0  # matching mode: manifest applies


# --------------------------------------------------------------------------- collator
class _CollProc:
    image_token_id = IMG_TOK
    def __init__(self):
        self.tokenizer = _Tok()


def _example(n_full, n_crop, n_cmp, text_len=300):
    fams = ["FULL"] * n_full + ["CROP"] * n_crop + ["COMPARE"] * n_cmp
    n_img_tok = sum(C.tokens_per_image(f) for f in fams)
    ids = torch.cat([torch.full((1, text_len), 5), torch.full((1, n_img_tok), IMG_TOK)], dim=1)
    labels = torch.full_like(ids, -100)
    labels[0, :20] = 7
    pix = torch.randn(sum(C.patches_per_image(f) for f in fams), C.PATCH_FEATURE_DIM)
    grid = torch.tensor([C.grid_thw(f) for f in fams], dtype=torch.long).reshape(-1, 3)
    return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == IMG_TOK).long(), "pixel_values": pix, "image_grid_thw": grid}


def test_static_vision_slot_padding_one_shape_for_every_mix_and_no_loss_leak():
    coll = BucketedQwenVLCollator(_CollProc(), max_seq_len=10240, pad_vision_to_slots=True)
    shapes = set()
    for mix in [(1, 0, 0), (1, 3, 1), (5, 10, 4), (2, 7, 0)]:
        ex = _example(*mix)
        out = coll([ex])
        shapes.add((tuple(out["pixel_values"].shape), tuple(out["image_grid_thw"].shape), tuple(out["input_ids"].shape)))
        assert out["pixel_values"].shape == (29952, 1536) and out["image_grid_thw"].shape == (19, 3)
        is_img = out["input_ids"] == IMG_TOK
        assert int(is_img.sum()) == 7488 == out["pixel_values"].shape[0] // 4  # tokens == features
        n_real = ex["input_ids"].shape[1]
        assert (out["attention_mask"][0, n_real:n_real + int((7488 - (ex["input_ids"] == IMG_TOK).sum()))] == 0).all()
        assert (out["labels"][0, n_real:] == -100).all()
        assert (out["labels"][0, :20] == 7).all()
        assert (out["pixel_values"][: ex["pixel_values"].shape[0]] == ex["pixel_values"]).all()  # real first
    assert len(shapes) == 1  # => exactly one XLA graph


def test_slot_padding_rejects_noncanonical_over_budget_and_overlength():
    coll = BucketedQwenVLCollator(_CollProc(), max_seq_len=10240, pad_vision_to_slots=True)
    bad = _example(1, 0, 0)
    bad["image_grid_thw"] = torch.tensor([[1, 10, 10]])
    with pytest.raises(ValueError, match="not a canonical family"):
        coll([bad])
    with pytest.raises(ValueError, match="static budget"):
        coll([_example(6, 0, 0)])
    with pytest.raises(ValueError, match="exceeds target length"):  # 7488 vision + 3000 text > 10240
        coll([_example(1, 0, 0, text_len=3000)])
    big = BucketedQwenVLCollator(_CollProc(), max_seq_len=16384, pad_vision_to_slots=True)
    assert big([_example(1, 0, 0, text_len=3000)])["input_ids"].shape == (1, 16384)


def test_gpu_path_untouched_by_default():
    coll = BucketedQwenVLCollator(_CollProc(), dynamic_padding=True)
    ex = _example(1, 2, 0)
    out = coll([ex])
    assert out["pixel_values"].shape == ex["pixel_values"].shape
    assert out["image_grid_thw"].shape == (3, 3)
    assert out["input_ids"].shape == ex["input_ids"].shape


# --------------------------------------------------------------------------- xla shim
def test_matmul_unit_lower_solve_matches_solve_triangular():
    torch.manual_seed(0)
    A = torch.randn(2, 3, 64, 64) * 0.1
    B = torch.randn(2, 3, 64, 32)
    ref = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
    assert torch.allclose(xp.unit_lower_solve_matmul(A, B), ref, atol=1e-4, rtol=1e-4)


def test_shim_only_intercepts_xla_unit_lower_and_is_idempotent():
    A = torch.randn(1, 8, 8) * 0.1
    B = torch.randn(1, 8, 4)
    real = torch.linalg.solve_triangular
    calls = []

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    torch.linalg.solve_triangular = counting  # shim will capture this as "original"
    try:
        assert xp.install_xla_solve_triangular_shim() is True
        assert xp.install_xla_solve_triangular_shim() is False
        torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)  # cpu -> original
        assert len(calls) == 1
        torch.linalg.solve_triangular(A, B, upper=True)  # other config -> original
        assert len(calls) == 2
        xp.SHIM_DEVICE_TYPES.add("cpu")
        out = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)  # intercepted
        assert len(calls) == 2
        assert torch.allclose(out, real(A, B, upper=False, unitriangular=True), atol=1e-5)
    finally:
        xp.SHIM_DEVICE_TYPES.discard("cpu")
        xp.uninstall_xla_solve_triangular_shim()
        torch.linalg.solve_triangular = real


# --------------------------------------------------------------------------- manifest lookup (regression)
def _write_traces(tmp_path, recs):
    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def test_manifest_written_by_compute_script_actually_filters(tmp_path):
    """Regression: compute_exact_trace_lengths.py writes '<file>::<dataset>::<id>' keys; the dataset
    used to look up only '<file>::<id>' / '<id>', so a fresh manifest matched nothing and filtered nothing."""
    tr = _write_traces(tmp_path, [
        {"image_id": 1, "dataset": "dentex", "messages": []},
        {"image_id": 2, "dataset": "dentex", "messages": []},
        {"image_id": 3, "messages": []},  # no 'dataset' -> compute script uses "default"
    ])
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "canonical_resize": False,
        "lengths_by_file_and_id": {"traces.jsonl::dentex::1": 5000, "traces.jsonl::dentex::2": 20000,
                                   "traces.jsonl::default::3": 30000},
        "lengths_by_image_id": {"dentex::1": 5000, "dentex::2": 20000, "default::3": 30000},
    }))
    ds = DentalSFTDataset(tr, processor=_RecordingProcessor(), max_seq_len=10240, token_lengths_manifest=manifest)
    assert [r["image_id"] for r in ds.records] == [1]


def test_slot_padded_length_filter_uses_text_plus_static_vision(tmp_path):
    tr = _write_traces(tmp_path, [
        {"image_id": 1, "dataset": "dentex", "messages": []},   # text 3000 + 1152 real vision
        {"image_id": 2, "dataset": "dentex", "messages": []},   # text 9500 + 1152 real vision
    ])
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "canonical_resize": True,
        "lengths_by_file_and_id": {"traces.jsonl::dentex::1": 4152, "traces.jsonl::dentex::2": 10652},
        "vision_tokens_by_file_and_id": {"traces.jsonl::dentex::1": 1152, "traces.jsonl::dentex::2": 1152},
    }))
    # raw lengths: trace 1 = 4152, trace 2 = 10652. Padded: 3000+7488=10488, 9500+7488=16988.
    common = dict(processor=_RecordingProcessor(), token_lengths_manifest=manifest, canonical_resize=True)
    raw = DentalSFTDataset(tr, max_seq_len=10240, **common)
    assert [r["image_id"] for r in raw.records] == [1]  # raw 10652 > 10240
    pad16 = DentalSFTDataset(tr, max_seq_len=16384, pad_vision_to_slots=True, **common)
    assert [r["image_id"] for r in pad16.records] == [1]  # 16988 > 16384, 10488 fits
    pad10 = DentalSFTDataset(tr, max_seq_len=10240, pad_vision_to_slots=True, **common)
    assert [r["image_id"] for r in pad10.records] == []  # 10488 > 10240: needs the 16384 bucket
    with pytest.warns(UserWarning, match="no per-trace vision-token counts"):
        m2 = tmp_path / "m2.json"
        m2.write_text(json.dumps({"canonical_resize": True,
                                  "lengths_by_file_and_id": {"traces.jsonl::dentex::1": 4152}}))
        DentalSFTDataset(tr, processor=_RecordingProcessor(), max_seq_len=10240, token_lengths_manifest=m2,
                         canonical_resize=True, pad_vision_to_slots=True)


def test_resolve_canonical_resize_env_and_explicit(monkeypatch):
    monkeypatch.delenv(C.ENV_CANONICAL_RESIZE, raising=False)
    assert C.resolve_canonical_resize(None) is False
    monkeypatch.setenv(C.ENV_CANONICAL_RESIZE, "1")
    assert C.resolve_canonical_resize(None) is True
    assert C.resolve_canonical_resize(False) is False  # explicit wins
    monkeypatch.setenv(C.ENV_CANONICAL_RESIZE, "0")
    assert C.resolve_canonical_resize(None) is False


# --------------------------------------------------------------------------- census
def test_census_family_counting_matches_dataset_attribution(tmp_path):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "census", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "census_vision_slots.py")
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)
    rec = _record(tmp_path, Image.new("RGB", (8, 8)))
    counts = census.count_families(rec)
    assert counts == {"FULL": 2, "CROP": 1, "COMPARE": 0}  # base + window_level=FULL, zoom_crop=CROP
