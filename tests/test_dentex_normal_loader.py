import unittest
from unittest.mock import patch
import pandas as pd
from dental_agent.data.dentex import load_dentex_normal_dataset

class TestDentexNormalLoader(unittest.TestCase):
    def test_dentex_normal_local_loading(self):
        """Test that local loading returns the 27 clinician-verified normal scans."""
        imgs_df, annots_df, cats_df = load_dentex_normal_dataset()
        self.assertEqual(len(imgs_df), 27)
        self.assertTrue(annots_df.empty)
        # Check known ground truth normal image IDs
        expected_ids = {3, 18, 36, 100, 173, 206, 214, 222, 263, 271, 274, 275, 297, 328, 355, 367, 372, 373, 406, 422, 434, 496, 587, 622, 626, 640, 666}
        self.assertEqual(set(imgs_df["id"]), expected_ids)

    def test_dentex_normal_merged_hf_json_simulation(self):
        """Simulate downloading the 1,339-image merged train.json from Hugging Face.
        In this merged file, all 1,339 images have tooth bounding boxes (category_id_1/2),
        but only 678 images have disease findings (category_id_3).
        """
        # Create 1,339 images
        simulated_imgs = pd.DataFrame([
            {"id": i, "file_name": f"{i}.png", "width": 1000, "height": 500, "local_path": f"/tmp/{i}.png"}
            for i in range(1, 1340)
        ])

        # Create tooth boxes for all 1,339 images
        tooth_annots = []
        for i in range(1, 1340):
            # All images have tooth boxes
            tooth_annots.append({
                "id": i,
                "image_id": i,
                "category_id_1": 1,
                "category_id_2": 1,
                "category_id_3": None, # No disease
                "bbox": [10, 10, 50, 50]
            })

        # Add disease findings for images 1..705 except the 27 normal ones
        normal_ids = {3, 18, 36, 100, 173, 206, 214, 222, 263, 271, 274, 275, 297, 328, 355, 367, 372, 373, 406, 422, 434, 496, 587, 622, 626, 640, 666}
        for i in range(1, 706):
            if i not in normal_ids:
                tooth_annots.append({
                    "id": 10000 + i,
                    "image_id": i,
                    "category_id_1": 1,
                    "category_id_2": 1,
                    "category_id_3": 0, # Caries
                    "bbox": [10, 10, 50, 50]
                })

        simulated_annots = pd.DataFrame(tooth_annots)
        simulated_cats = pd.DataFrame([{"id": 0, "name": "Caries"}])

        with patch("dental_agent.data.dentex.load_dentex_dataset", return_value=(simulated_imgs, simulated_annots, simulated_cats)):
            normal_imgs, empty_annots, _ = load_dentex_normal_dataset()
            self.assertEqual(len(normal_imgs), 27, f"Expected 27 normal images from merged HF json, got {len(normal_imgs)}")
            self.assertEqual(set(normal_imgs["id"]), normal_ids)
            self.assertTrue(empty_annots.empty)

if __name__ == "__main__":
    unittest.main()
