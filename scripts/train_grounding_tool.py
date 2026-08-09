import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Grounding Tool on DENTEX.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on (e.g., '0' for GPU 0, 'cpu' for CPU)")
    args = parser.parse_args()

    yaml_path = Path("data/yolo_dentex/dataset.yaml").absolute()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cannot find {yaml_path}. Run prepare_yolo_dataset.py first.")

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on {args.device} for {args.epochs} epochs...")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="data/models",
        name="grounding_tool",
        exist_ok=True # Overwrite existing training run if named the same
    )
    
    print("\nTraining Complete!")
    print(f"Model saved to: data/models/grounding_tool/weights/best.pt")

if __name__ == "__main__":
    main()
