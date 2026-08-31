import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import tempfile
import os

from dental_agent.data.tufts import find_local_tufts_dir, _is_valid_tufts_root
from scripts.prepare_yolo_dataset import convert_single_image, _ensure_images_downloaded, create_dataset_yaml

class TestPrepareYOLODataset(unittest.TestCase):
    def test_tufts_ignores_yolo_output_directories(self):
        """find_local_tufts_dir must strictly reject YOLO output folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yolo_fake = Path(tmpdir) / "yolo_dentex_tufts_cv"
            yolo_fake.mkdir()
            self.assertFalse(_is_valid_tufts_root(yolo_fake))

            # A real Tufts directory should have Segmentation/teeth_bbox.json or Radiographs
            real_fake = Path(tmpdir) / "Tufts"
            (real_fake / "Segmentation").mkdir(parents=True)
            (real_fake / "Segmentation" / "teeth_bbox.json").write_text("{}", encoding="utf-8")
            self.assertTrue(_is_valid_tufts_root(real_fake))

    def test_convert_single_image_yolo_format(self):
        """convert_single_image must output correct 0-31 class index and normalized coords."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img_file = tmp / "1.png"
            img_file.write_bytes(b"dummy")

            images_out = tmp / "images"
            labels_out = tmp / "labels"
            images_out.mkdir()
            labels_out.mkdir()

            img_row = {
                "id": 1,
                "local_path": str(img_file),
                "width": 1000,
                "height": 500,
            }
            # FDI 18 -> class_idx = (1-1)*8 + (8-1) = 7
            annots_df = pd.DataFrame([{
                "image_id": 1,
                "category_id_1": 0, # 0-indexed DENTEX format
                "category_id_2": 7,
                "bbox": [100, 50, 200, 100], # x_min, y_min, w, h
            }])

            from dental_agent.data.dentex import dentex_row_to_fdi
            res = convert_single_image(img_row, annots_df, images_out, labels_out, quadrant_position_fn=dentex_row_to_fdi, dataset_tag="dentex")
            self.assertIsNotNone(res)

            label_file = labels_out / f"{res}.txt"
            self.assertTrue(label_file.exists())
            content = label_file.read_text().strip()
            parts = content.split()
            self.assertEqual(parts[0], "7") # Class 7 for tooth 18
            self.assertAlmostEqual(float(parts[1]), (100 + 100) / 1000.0) # x_center = 200/1000 = 0.2
            self.assertAlmostEqual(float(parts[2]), (50 + 50) / 500.0)    # y_center = 100/500 = 0.2

    def test_ensure_images_downloaded_from_hf_slice(self):
        """_ensure_images_downloaded must populate local_path from paths_map returned by download_slice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_img = Path(tmpdir) / "10.png"
            fake_img.write_bytes(b"dummy")

            images_df = pd.DataFrame([{"id": 10, "local_path": None}])
            with patch.dict(os.environ, {"DENTEX_IMAGES_REPO": "dummy/repo"}):
                with patch("dental_agent.data.dentex.download_dentex_slice", return_value={10: fake_img}):
                    res_df = _ensure_images_downloaded(images_df, "dentex", data_dir=tmpdir)
                    self.assertIsNotNone(res_df.iloc[0]["local_path"])
                    self.assertEqual(res_df.iloc[0]["local_path"], str(fake_img.resolve()))

    def test_create_dataset_yaml(self):
        """create_dataset_yaml must create valid YAML with 32 tooth classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            create_dataset_yaml(out_dir, val_subdir="images/val")
            yaml_path = out_dir / "dataset.yaml"
            self.assertTrue(yaml_path.exists())
            text = yaml_path.read_text()
            self.assertIn("Tooth 11", text)
            self.assertIn("Tooth 48", text)

    def test_dataset_dir_suffix_and_model_prefix(self):
        """_dataset_dir_suffix and _model_subdir_prefix must handle comma and space-separated multi-datasets."""
        from scripts.train_grounding_tool import _dataset_dir_suffix, _model_subdir_prefix
        self.assertEqual(_dataset_dir_suffix("dentex"), "dentex")
        self.assertEqual(_dataset_dir_suffix("dentex,tufts"), "dentex_tufts")
        self.assertEqual(_dataset_dir_suffix("dentex tufts"), "dentex_tufts")

        self.assertEqual(_model_subdir_prefix("dentex"), "")
        self.assertEqual(_model_subdir_prefix("dentex_tufts"), "dentex_tufts_")

if __name__ == "__main__":
    unittest.main()
