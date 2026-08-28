from pathlib import Path
from PIL import Image
import pandas as pd

from dental_agent.training.trace_generation import resolve_trace_image_path


def test_resolve_trace_image_path_direct_exists(tmp_path):
    img_file = tmp_path / "4.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_file)

    resolved = resolve_trace_image_path(str(img_file), image_id=4)
    assert resolved is not None
    assert resolved.exists()
    assert resolved == img_file


def test_resolve_trace_image_path_cross_environment_fallback(tmp_path):
    # Simulate a Kaggle path that does NOT exist on the current system
    fake_kaggle_path = "/kaggle/working/dental_agent/data/datasets--Reza-Nadimi--dentex-train-images/snapshots/5e30580c/images/4.png"

    # But the local environment has the image in a local data_dir
    local_img_dir = tmp_path / "dentex" / "training_data" / "images"
    local_img_dir.mkdir(parents=True, exist_ok=True)
    local_img_file = local_img_dir / "4.png"
    Image.new("RGB", (100, 100), color="green").save(local_img_file)

    resolved = resolve_trace_image_path(
        fake_kaggle_path,
        image_id=4,
        dataset_name="dentex",
        data_dir=tmp_path,
    )
    assert resolved is not None
    assert resolved.exists()
    assert resolved.name == "4.png"


def test_resolve_trace_image_path_via_images_df(tmp_path):
    fake_path = "/non_existent_mount/val_15.png"
    real_img = tmp_path / "actual_val_15.png"
    Image.new("RGB", (100, 100), color="red").save(real_img)

    images_df = pd.DataFrame([{"id": 15, "local_path": str(real_img)}])

    resolved = resolve_trace_image_path(
        fake_path,
        image_id=15,
        images_df=images_df,
    )
    assert resolved is not None
    assert resolved == real_img


def test_resolve_trace_image_path_dynamic_slice_download(monkeypatch, tmp_path):
    fake_path = "/remote_env/images/999999.png"
    downloaded_img = tmp_path / "999999.png"
    Image.new("RGB", (100, 100), color="yellow").save(downloaded_img)

    monkeypatch.setenv("DENTEX_IMAGES_REPO", "mock/dentex-images")
    monkeypatch.setattr(
        "dental_agent.data.dentex.download_dentex_slice",
        lambda ids, **kwargs: {999999: downloaded_img},
    )

    resolved = resolve_trace_image_path(
        fake_path,
        image_id=999999,
        dataset_name="dentex",
    )
    assert resolved is not None
    assert resolved == downloaded_img
