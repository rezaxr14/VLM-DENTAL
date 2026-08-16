"""
Test the LangGraph-based trace generation loop against a local vLLM instance.
Assumes vLLM is running locally and serving Qwen/Qwen3-VL-8B-Thinking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from PIL import Image

from dental_agent.config import load_env
from dental_agent.tools.registry import ToolRegistry
from dental_agent.agent.prompts import generate_agent_prompt
from dental_agent.agent.langgraph_loop import run_trace_gen

def main():
    parser = argparse.ArgumentParser(description="Test LangGraph loop with vLLM")
    parser.add_argument("--image", required=True, help="Path to input X-ray image")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking", help="Model name in vLLM")
    args = parser.parse_args()

    load_env()
    
    # Setup tools
    registry = ToolRegistry()
    registry.register_all()
    
    # Generate prompt
    system_prompt = generate_agent_prompt(registry)
    
    print(f"Loading image from {args.image}")
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Failed to load image: {e}")
        sys.exit(1)
        
    # Dummy ground truth for testing
    ground_truth = [{
        "quadrant": 3,
        "tooth_position": 6,
        "diagnosis": "Caries",
        "bbox": [100, 100, 200, 200]
    }]
    
    print(f"\n--- Testing LangGraph Loop against {args.model} ---")
    print("This will execute real tool functions and pass results back to the model.")
    
    t0 = time.time()
    result, error = run_trace_gen(
        image=image,
        ground_truth=ground_truth,
        registry=registry,
        system_prompt=system_prompt,
        provider="local",
        model=args.model,
        max_turns=5
    )
    t1 = time.time()
    
    if error:
        print(f"\n[ERROR] Loop failed after {t1-t0:.2f}s: {error}")
    elif result:
        print(f"\n[SUCCESS] Loop completed in {t1-t0:.2f}s with {result['tool_calls']} tool calls.")
        print("\nFinal Answer:")
        print(json.dumps(result["final_answer"], indent=2))
        
        print("\nTrajectory turns:")
        for t in result["turns"]:
            if "tool_name" in t:
                print(f" - Turn {t['turn']}: Called {t['tool_name']} with args {t['tool_args']}")
                if not t.get("tool_ok"):
                    print(f"   Tool error: {t.get('tool_error')}")
            elif "status" in t and t["status"] == "final_answer":
                print(f" - Turn {t['turn']}: Generated final answer")

if __name__ == "__main__":
    main()
