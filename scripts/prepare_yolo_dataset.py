import os
import shutil
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm

from dental_agent.data.dentex import load_dentex_dataset

def convert_to_yolo_format(output_dir: str | Path, split: str = "train", data_dir: str = "data"):
    """
    Convert DENTEX COCO annotations to YOLOv8 txt format.
    Output structure:
        output_dir/
            images/
                train/
                val/
            labels/
                train/
                val/
            dataset.yaml
    """
    output_dir = Path(output_dir)
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {split} split from DENTEX...")
    try:
        images_df, annots_df, _ = load_dentex_dataset(data_dir=data_dir, split_name=split)
    except Exception as e:
        print(f"Failed to load split '{split}': {e}")
        return

    # Filter to only images that exist locally
    valid_images = images_df[images_df["local_path"].notna()].copy()
    
    print(f"Found {len(valid_images)} valid images. Processing...")
    
    for _, img_row in tqdm(valid_images.iterrows(), total=len(valid_images)):
        img_id = img_row["id"]
        local_path = Path(img_row["local_path"])
        
        if not local_path.exists():
            continue
            
        img_w = img_row["width"]
        img_h = img_row["height"]
        
        # Get annotations for this image
        img_annots = annots_df[annots_df["image_id"] == img_id]
        
        yolo_lines = []
        for _, ann in img_annots.iterrows():
            if pd.isna(ann.get("category_id_1")) or pd.isna(ann.get("category_id_2")):
                continue
                
            quadrant = int(ann["category_id_1"])
            position = int(ann["category_id_2"])
            
            # Map FDI (Quadrant 1-4, Position 1-8) to YOLO Class (0-31)
            # Class 0 = Tooth 11, Class 31 = Tooth 48
            if not (1 <= quadrant <= 4 and 1 <= position <= 8):
                continue
                
            class_idx = (quadrant - 1) * 8 + (position - 1)
            
            # COCO bbox: [x_min, y_min, width, height]
            bbox = ann["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
                
            x_min, y_min, bw, bh = bbox
            
            # YOLO format: normalized x_center, y_center, width, height
            x_center = (x_min + bw / 2.0) / img_w
            y_center = (y_min + bh / 2.0) / img_h
            w_norm = bw / img_w
            h_norm = bh / img_h
            
            # Clamp values between 0 and 1
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        if not yolo_lines:
            continue # Skip images with no valid tooth annotations
            
        # Copy image to YOLO directory
        dest_img_path = images_out / local_path.name
        if not dest_img_path.exists():
            shutil.copy2(local_path, dest_img_path)
            
        # Write YOLO label file
        label_path = labels_out / f"{local_path.stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
            
def create_dataset_yaml(output_dir: str | Path):
    """Generate the dataset.yaml file required by ultralytics YOLO."""
    output_dir = Path(output_dir)
    
    # Generate names dictionary: {0: "Tooth 11", 1: "Tooth 12", ...}
    names = {}
    for q in range(1, 5):
        for p in range(1, 9):
            idx = (q - 1) * 8 + (p - 1)
            names[idx] = f"Tooth {q}{p}"
            
    yaml_content = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/validation", # We use validation_data for val
        "names": names
    }
    
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

if __name__ == "__main__":
    yolo_dir = Path("data/yolo_dentex")
    
    print("Preparing YOLO Dataset...")
    convert_to_yolo_format(yolo_dir, split="validation")
    convert_to_yolo_format(yolo_dir, split="train")
    
    create_dataset_yaml(yolo_dir)
    print(f"\nDone! Dataset ready at {yolo_dir}/dataset.yaml")
