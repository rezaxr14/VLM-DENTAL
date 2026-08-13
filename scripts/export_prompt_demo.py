import argparse
import json
import os
import sys
import pandas as pd
from PIL import Image

# Add project root to path so we can import dental_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dental_agent.agent.prompts import build_agent_system_prompt
from dental_agent.tools.registry import ToolRegistry
from dental_agent.tools.windowing import tool_window_level
from dental_agent.tools.denoise import tool_denoise
from dental_agent.tools.zoom_crop import tool_zoom_crop
from dental_agent.tools.contralateral import tool_contralateral_compare
from dental_agent.tools.grounding import tool_locate_tooth

def main():
    parser = argparse.ArgumentParser(description="Export a full prompt and tool execution demo for a specific DENTEX image.")
    parser.add_argument("--image-id", type=int, default=34, help="The DENTEX image ID to generate a demo for.")
    parser.add_argument("--output-dir", type=str, default="examples/prompt_demos", help="Base directory to save the demos.")
    args = parser.parse_args()

    image_id = args.image_id
    out_dir = os.path.join(args.output_dir, f"image_{image_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating prompt demo for Image ID {image_id}...")
    
    # Paths (relative to project root)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    labels_path = os.path.join(root_dir, "data", "dentex", "DENTEX", "training_data", "quadrant-enumeration-disease", "train_quadrant_enumeration_disease.json")
    
    if not os.path.exists(labels_path):
        print(f"Error: Could not find DENTEX labels at {labels_path}")
        print("Please ensure you are running this from the project root or the data directory is set up correctly.")
        sys.exit(1)

    # Load data
    with open(labels_path, 'r') as f:
        annots_json = json.load(f)

    images_df = pd.DataFrame(annots_json.get('images', []))
    annotations_df = pd.DataFrame(annots_json.get('annotations', []))
    categories_df = pd.DataFrame(annots_json.get('categories', []))
    cat_lookup = dict(zip(categories_df['id'], categories_df['name'])) if not categories_df.empty else {}

    # Get Image data
    anns = annotations_df[annotations_df['image_id'] == image_id]
    if anns.empty:
        print(f"Error: No annotations found for image ID {image_id}.")
        sys.exit(1)
        
    image_rows = images_df[images_df['id'] == image_id]
    if image_rows.empty:
        print(f"Error: No image record found for image ID {image_id}.")
        sys.exit(1)
        
    image_filename = image_rows['file_name'].iloc[0]
    image_path = os.path.join(root_dir, "data", "yolo_dentex", "images", "train", image_filename)
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image file at {image_path}")
        sys.exit(1)

    base_img = Image.open(image_path)
    base_img.save(os.path.join(out_dir, '0_original.jpg'))

    # Load tools
    registry = ToolRegistry.create_default()

    # 1. Windowing
    print("Applying windowing tools...")
    tool_window_level(base_img, 'bone').save(os.path.join(out_dir, '1_window_bone.jpg'))
    tool_window_level(base_img, 'enamel').save(os.path.join(out_dir, '2_window_enamel.jpg'))
    tool_window_level(base_img, 'soft_tissue').save(os.path.join(out_dir, '3_window_soft_tissue.jpg'))

    # 2. Denoising
    print("Applying denoising tools...")
    tool_denoise(base_img, 'bilateral').save(os.path.join(out_dir, '4_denoise_bilateral.jpg'))
    tool_denoise(base_img, 'median').save(os.path.join(out_dir, '5_denoise_median.jpg'))

    # Format GT & Execute Crops
    print("Pre-computing dynamic bounding box tools...")
    findings = []
    crop_index = 6
    for _, ann in anns.iterrows():
        diag_id = ann.get('category_id_3')
        fallback_map = {0: 'Impacted', 1: 'Caries', 2: 'Periapical Lesion', 3: 'Deep Caries'}
        diag_name = cat_lookup.get(diag_id) or fallback_map.get(diag_id, 'unknown')
        fdi_quadrant = int(ann.get('category_id_1', 0)) + 1
        fdi_position = int(ann.get('category_id_2', 0)) + 1
        bbox = list(ann.get('bbox', [0, 0, 50, 50]))
        
        findings.append({
            'quadrant': fdi_quadrant,
            'tooth_position': fdi_position,
            'diagnosis': diag_name,
            'bbox': bbox
        })
        
        # Execute zoom crop for this finding
        tool_zoom_crop(base_img, bbox).save(os.path.join(out_dir, f'{crop_index}_crop_tooth_{fdi_quadrant}{fdi_position}.jpg'))
        crop_index += 1
        
        # Execute contralateral compare for this finding
        # We wrap in try-except in case bbox is invalid for contralateral
        try:
            tool_contralateral_compare(base_img, bbox, fdi_quadrant).save(os.path.join(out_dir, f'{crop_index}_contralateral_quad_{fdi_quadrant}.jpg'))
            crop_index += 1
        except Exception as e:
            print(f"Skipping contralateral for bbox {bbox}: {e}")
            
        # Execute locate_tooth for this finding
        fdi_num = int(f"{fdi_quadrant}{fdi_position}")
        try:
            loc_result = tool_locate_tooth(base_img, fdi_num)
            with open(os.path.join(out_dir, f'{crop_index}_locate_tooth_{fdi_num}.json'), 'w') as lf:
                json.dump(loc_result, lf)
            crop_index += 1
        except Exception as e:
            print(f"Skipping locate_tooth for {fdi_num}: {e}")


    # Create prompt text
    print("Generating exact LLM prompt...")
    sys_prompt = build_agent_system_prompt(registry.format_tool_descriptions())
    
    directive = (
        "TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.\n"
        f"You MUST eventually reach this exact diagnosis: {json.dumps(findings)}\n\n"
        "The ground-truth findings above tell you what's there and roughly where — use that "
        "as your starting hint for where to look, not as something to restate without checking. "
        "Use zoom_crop / window_level / denoise / contralateral_compare / locate_tooth for real "
        "to inspect each region before your final answer. You MUST use at least one tool before "
        "answering — do not output final_answer on the first turn."
    )
    
    user_prompt = f"Analyze this panoramic X-ray. Identify any abnormal teeth and determine the diagnosis.\n\n{directive}"

    with open(os.path.join(out_dir, 'prompt.md'), 'w') as f:
        f.write("# System Prompt\n```text\n" + sys_prompt + "\n```\n\n# User Prompt\n```text\n" + user_prompt + "\n```\n")

    print(f"\n✅ Success! Demo created at: {out_dir}")
    print(f"You can now share this directory to demonstrate the Agentic CoT generation process.")

if __name__ == "__main__":
    main()
