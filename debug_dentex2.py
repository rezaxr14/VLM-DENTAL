from dental_agent.data.dentex import discover_annotation_files, pick_best_annotation_file

all_coco = discover_annotation_files("data")
print("Total JSONs:", len(all_coco))
candidates = [
    (jf, d) for jf, d in all_coco.items()
    if isinstance(d, dict) and d.get("annotations")
]
print("Candidates with annotations:", len(candidates))
best = pick_best_annotation_file(all_coco, "train")
print("Best:", best[0])
