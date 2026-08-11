"""
Unit tests for combining multiple DENTEX dataset folders (quadrant-enumeration + quadrant-enumeration-disease)
for YOLO training.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dental_agent.data.dentex import load_combined_dentex_dataset
from scripts.prepare_yolo_dataset import convert_to_yolo_format


def mock_dentex_data(tmp_path: Path):
    """Creates a temporary mock DENTEX structure with 2 dataset subfolders."""
    dentex_root = tmp_path / "data" / "dentex" / "DENTEX"
    train_root = dentex_root / "training_data"
    
    folder1 = train_root / "quadrant-enumeration-disease"
    folder1_xrays = folder1 / "xrays"
    folder1_xrays.mkdir(parents=True, exist_ok=True)
    
    # Create fake image files
    img1_file = folder1_xrays / "train_1.png"
    img1_file.write_bytes(b"fake image 1")
    
    coco1 = {
        "images": [{"id": 1, "file_name": "xrays/train_1.png", "width": 1000, "height": 500}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id_1": 1,
                "category_id_2": 6,
                "category_id_3": 1,
                "bbox": [100, 100, 50, 50],
            }
        ],
        "categories": [{"id": 1, "name": "Caries"}],
    }
    with open(folder1 / "train_quadrant_enumeration_disease.json", "w") as f:
        json.dump(coco1, f)
        
    folder2 = train_root / "quadrant-enumeration"
    folder2_xrays = folder2 / "xrays"
    folder2_xrays.mkdir(parents=True, exist_ok=True)
    
    img2_file = folder2_xrays / "train_2.png"
    img2_file.write_bytes(b"fake image 2")
    
    coco2 = {
        "images": [{"id": 1, "file_name": "xrays/train_2.png", "width": 1000, "height": 500}],
        "annotations": [
            {
                "id": 20,
                "image_id": 1,
                "category_id_1": 3,
                "category_id_2": 8,
                "bbox": [200, 200, 60, 60],
            }
        ],
        "categories": [{"id": 1, "name": "Tooth"}],
    }
    with open(folder2 / "train_quadrant_enumeration.json", "w") as f:
        json.dump(coco2, f)
        
    return dentex_root


def test_load_combined_dentex_dataset(mock_dentex_path: Path):
    images_df, annots_df, _ = load_combined_dentex_dataset(
        data_dir=str(mock_dentex_path.parent.parent),
        split_name="train",
        use_cache=False,
    )
    
    assert len(images_df) == 2, f"Expected 2 images, got {len(images_df)}"
    assert len(annots_df) == 2, f"Expected 2 annotations, got {len(annots_df)}"
    # Ensure IDs were re-indexed uniquely
    assert list(images_df["id"]) == [1, 2], f"Expected IDs [1, 2], got {list(images_df['id'])}"
    assert list(annots_df["image_id"]) == [1, 2], f"Expected image_ids [1, 2], got {list(annots_df['image_id'])}"


def test_convert_to_yolo_format(mock_dentex_path: Path, tmp_path: Path):
    output_yolo = tmp_path / "yolo_output"
    
    convert_to_yolo_format(
        output_dir=output_yolo,
        split="train",
        data_dir=str(mock_dentex_path.parent.parent),
    )
    
    train_images = list((output_yolo / "images" / "train").glob("*"))
    train_labels = list((output_yolo / "labels" / "train").glob("*.txt"))
    
    assert len(train_images) == 2, f"Expected 2 yolo images, got {len(train_images)}"
    assert len(train_labels) == 2, f"Expected 2 yolo labels, got {len(train_labels)}"


if __name__ == "__main__":
    import shutil
    tmp = Path("tmp_test_run")
    tmp.mkdir(exist_ok=True)
    try:
        data_path = mock_dentex_data(tmp)
        test_load_combined_dentex_dataset(data_path)
        test_convert_to_yolo_format(data_path, tmp)
        print("ALL TESTS PASSED SUCCESSFULLY!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
