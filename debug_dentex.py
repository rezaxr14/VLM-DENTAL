from dental_agent.data.dentex import discover_annotation_files
import glob

# Try loading all
all_coco = discover_annotation_files("data")
print(f"Total loaded: {len(all_coco)}")
for k in all_coco.keys():
    if "train_quadrant" in k:
        print("Found:", k)
