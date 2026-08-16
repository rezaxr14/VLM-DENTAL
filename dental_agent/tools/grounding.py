import os
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ToothGrounder:
    """Singleton wrapper for the YOLOv8 grounding model to avoid reloading weights repeatedly."""
    _instance = None
    
    def __init__(self, model_path: str | None = None):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

        # NOTE: grounding tool is YOLOv8m trained with 5-fold cross-validation (not YOLOv8x).
        # Decision: uses a single best-performing fold's weights, not an ensemble across all 5 —
        # simpler and cheaper at inference time, and the single-fold validation numbers
        # (mAP50 ≈ 0.647, R ≈ 0.90, P ≈ 0.588) already clear the bar this tool needs to hit.
        # Point GROUNDING_MODEL_PATH at that fold's best.pt.
        self.model_path = Path(
            model_path
            or os.environ.get("GROUNDING_MODEL_PATH", "data/models/grounding_tool_cv_best/weights/best.pt")
        )
        self.model = None
        
        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
            
    @classmethod
    def get_instance(cls) -> "ToothGrounder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def locate_tooth(self, image: Image.Image, fdi_number: int) -> Dict[str, Any]:
        if self.model is None:
            return {"error": f"Grounding model not found at {self.model_path}. Train it first using scripts/train_grounding_tool.py"}
            
        # Convert FDI to YOLO class index
        # Format: quadrant (1-4) and position (1-8)
        fdi_str = str(fdi_number)
        if len(fdi_str) != 2:
            return {"error": f"Invalid FDI number format: {fdi_number}. Must be two digits (e.g. 38)."}
            
        try:
            quadrant = int(fdi_str[0])
            position = int(fdi_str[1])
        except ValueError:
            return {"error": f"Invalid FDI number format: {fdi_number}"}
            
        if not (1 <= quadrant <= 4 and 1 <= position <= 8):
            return {"error": f"Invalid FDI quadrant or position: {fdi_number}"}
            
        target_class_idx = (quadrant - 1) * 8 + (position - 1)
        
        # Run inference
        results = self.model.predict(image, verbose=False, conf=0.25)
        
        if not results or len(results) == 0:
            return {"error": "Model failed to return any predictions."}
            
        result = results[0]
        boxes = result.boxes
        
        # Find the box with the target class index and highest confidence
        best_box = None
        best_conf = -1.0
        
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            
            if cls_idx == target_class_idx and conf > best_conf:
                best_box = boxes.xywh[i] # Center format
                best_conf = conf
                
        if best_box is None:
            return {"error": f"Tooth {fdi_number} not found in this radiograph."}
            
        # Convert xywh (center format) to xywh (top-left format for crop/zoom tools)
        cx, cy, w, h = best_box.tolist()
        x_min = cx - (w / 2)
        y_min = cy - (h / 2)
        
        return {
            "tooth": fdi_number,
            "confidence": round(best_conf, 3),
            "bbox": [round(x_min, 1), round(y_min, 1), round(w, 1), round(h, 1)]
        }

def tool_locate_tooth(image: Image.Image, tooth: int | str) -> Dict[str, Any]:
    """
    Locates a specific tooth using a trained YOLOv8 model and returns its bounding box.

    Args:
        image: The panoramic radiograph (PIL.Image)
        tooth: The FDI tooth number to locate (e.g., 38 for lower-left 3rd molar).

    Returns:
        Dictionary containing the bounding box [x, y, w, h] of the tooth, or an error message.

    NOTE: this signature was previously `(image, args: Dict[str, Any])`, which does not match
    how ToolRegistry.execute(name, **kwargs) actually calls tools (it passes flat keyword
    arguments, e.g. tool=38, not a nested `args` dict) — every other tool in this module
    already takes flat kwargs, so this was a real bug: any real call to locate_tooth through
    the registry raised a TypeError, silently caught and reported as "tool execution failed"
    by the caller. Fixed to match the flat-kwargs convention used everywhere else.
    """
    if tooth is None:
        return {"error": "Missing required argument 'tooth' (FDI number)."}

    try:
        tooth = int(tooth)
    except (TypeError, ValueError):
        return {"error": "Argument 'tooth' must be an integer (e.g., 38)."}

    grounder = ToothGrounder.get_instance()
    return grounder.locate_tooth(image, tooth)
