import os
import shutil
from pathlib import Path
from PIL import Image

# Clear out any existing model in the default download path
default_path = Path("data/models/yolo_cv_best/weights/best.pt")
if default_path.exists():
    print(f"Test setup: Removing existing model at {default_path}")
    default_path.unlink()

# Unset the env var so it relies on the fallback
if "GROUNDING_MODEL_PATH" in os.environ:
    del os.environ["GROUNDING_MODEL_PATH"]

# Ensure HF_ARTIFACT_REPO is set correctly (just in case)
os.environ["HF_ARTIFACT_REPO"] = "Reza-Nadimi/vlm-dental-models"

print("--- Running locate_tooth with missing model ---")

import sys
sys.path.append(os.getcwd())
try:
    from dental_agent.tools.grounding import tool_locate_tooth, ToothGrounder
    
    # We must reset the singleton instance to force a fresh __init__ 
    # since it might have been loaded in a different state if imported earlier.
    ToothGrounder._instance = None
    
    # Create a dummy image
    img = Image.new('RGB', (1000, 1000), color='black')
    
    # Locate tooth 38
    print("Invoking locate_tooth for FDI 38...")
    result = tool_locate_tooth(img, 38)
    
    print("\nResult:")
    print(result)
    
    if "error" in result and "Grounding model not found" in result["error"]:
        print("[FAIL] TEST FAILED: Model was not downloaded!")
        exit(1)
    
    if default_path.exists():
        print("[PASS] TEST PASSED: Model was successfully auto-downloaded!")
    else:
        print("[FAIL] TEST FAILED: Model was not saved to expected path.")
        exit(1)

except Exception as e:
    print(f"[FAIL] TEST FAILED with exception: {e}")
    exit(1)
