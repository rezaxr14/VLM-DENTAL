# VLM-DENTAL Useful Commands Snippet

Here are the most frequently used commands to run various parts of the VLM-DENTAL pipeline locally.

> [!NOTE]
> Ensure your virtual environment is activated and you are in the root directory (`VLM-DENTAL`) before running these.

## 1. Trace Generation
Generate the synthetic CoT trajectories using the Gemini API.
```powershell
# Run the generator and save output to the traces directory
python scripts/run_daily_trace_generator.py --split train --output data/traces/train_cot_traces.jsonl
```

## 2. YOLO Grounding Tool
Convert the DENTEX annotations and train the bounding-box model.
```powershell
# Step 1: Convert COCO annotations to YOLO format
python scripts/prepare_yolo_dataset.py

# Step 2: Train YOLOv8 Medium (requires a GPU)
yolo train data=data/yolo_dentex/dataset.yaml model=yolov8m.pt epochs=500 imgsz=640 batch=16 device=0 project=data/models name=grounding_tool
```

## 3. Supervised Fine-Tuning (SFT)
Fine-tune Qwen-VL on your generated traces.
```powershell
python scripts/train_sft.py --dataset_path data/traces/train_cot_traces.jsonl --output_dir data/models/qwen3_vl_sft --batch_size 1 --epochs 3
```

## 4. Reinforcement Learning (GRPO)
Optimize the model's tool usage using PEFT multi-adapter GRPO.
```powershell
python scripts/run_grpo.py --sft-model-dir data/models/qwen3_vl_sft --group-size 4 --epochs 2
```

## 5. Setup & Misc
```powershell
# Download and extract the raw dataset
python scripts/download_dataset.py

# Install project dependencies
pip install -e .
pip install python-dotenv pandas pillow google-generativeai anthropic huggingface_hub ultralytics trl peft bitsandbytes
```
