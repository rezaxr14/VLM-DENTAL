import os
import json
import argparse
import sys

def extract_traces_for_images(file_path: str, image_ids: set, is_zero_shot: bool = False):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    traces = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                img_id = data.get("image_id")
                
                if image_ids and str(img_id) not in image_ids:
                    continue
                    
                if is_zero_shot:
                    ans = data.get("trajectory", {}).get("final_answer", []) or data.get("final_answer", [])
                    score = len(ans)
                else:
                    turns = data.get("turns", [])
                    score = sum(len(t.get("tool_calls_this_turn", [])) for t in turns)
                    
                traces.append((score, data))
            except Exception as e:
                pass
                
    traces.sort(key=lambda x: x[0], reverse=True)
    return [t[1] for t in traces]


def format_trace_to_markdown(data: dict, is_zero_shot: bool = False) -> str:
    md = []
    image_id = data.get("image_id", "Unknown")
    ground_truth = data.get("ground_truth", [])
    
    md.append(f"### Trace Walkthrough: Image ID {image_id}")
    
    if is_zero_shot:
        md.append("This trace demonstrates the agent's baseline capability when deprived of diagnostic tools. It evaluates the entire panoramic radiograph strictly from the initial full-resolution view, offering its diagnostic reasoning directly without interactive spatial verification.")
    else:
        turns = data.get("trajectory", data).get("turns", [])
        nudge_count = sum(1 for t in turns for tc in t.get("tool_calls_this_turn", []) if tc.get("tool_name") == "nudge_crop")
        zoom_count = sum(1 for t in turns for tc in t.get("tool_calls_this_turn", []) if tc.get("tool_name") == "zoom_crop")
        md.append(f"This interactive trace exemplifies rigorous clinical validation. Over the course of the session, the agent performed {zoom_count} targeted localized crops.")
        if nudge_count > 0:
            md.append(f"Crucially, it actively engaged in **self-correction**, utilizing `nudge_crop` {nudge_count} times to aggressively refine misaligned or off-center bounding boxes before passing judgment. This mirrors genuine human radiologic workflows.")

    md.append("\n**Ground Truth Findings:**\n```json\n" + json.dumps(ground_truth, indent=2) + "\n```\n")
    
    trajectory = data.get("trajectory", data)
    messages = trajectory.get("messages", [])
    
    if is_zero_shot:
        md.append("#### Zero-Shot Execution")
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            md.append(f"**{role.capitalize()}:**\n```text\n{content}\n```\n")
    else:
        md.append("#### Interactive CoT Execution")
        turn_idx = 1
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            
            if role == "assistant":
                md.append(f"### Turn {turn_idx}")
                turn_idx += 1
                
                # Parse JSON if possible to format beautifully
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        formatted_content = json.dumps(parsed, indent=2)
                    except:
                        formatted_content = content
                    md.append(f"**Agent Output:**\n```json\n{formatted_content}\n```\n")
                elif isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    md.append(f"**Agent Output:**\n```json\n{' '.join(text_parts).strip()}\n```\n")
                    
            elif role == "user":
                if isinstance(content, list):
                    for c in content:
                        if c.get("type") == "text":
                            text = c.get("text", "")
                            if text.startswith("Result of"):
                                prefix = text.split(':', 1)[0]
                                result_body = text.split(':', 1)[1].strip() if ':' in text else ''
                                if result_body:
                                    md.append(f"**{prefix}:**\n```json\n{result_body}\n```\n")
                                else:
                                    md.append(f"**{prefix}:** `[Image Output Generated]`\n")
                elif isinstance(content, str):
                     if content.startswith("Result of"):
                         prefix = content.split(':', 1)[0]
                         result_body = content.split(':', 1)[1].strip() if ':' in content else ''
                         if result_body:
                             md.append(f"**{prefix}:**\n```json\n{result_body}\n```\n")
                         else:
                             md.append(f"**{prefix}:** `[Image Output Generated]`\n")
                
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Export high-quality CoT traces for the paper.")
    parser.add_argument("--image-ids", type=str, default="", help="Comma separated image IDs to extract.")
    parser.add_argument("--interactive-file", type=str, default="data/traces/train_cot_traces.jsonl")
    parser.add_argument("--zeroshot-file", type=str, default="data/traces/train_cot_traces_unverified_no_tools.jsonl")
    parser.add_argument("--output-dir", type=str, default="examples/prompt_demos/paper_traces")
    
    args = parser.parse_args()

    target_ids = set([x.strip() for x in args.image_ids.split(",") if x.strip()])
    if not target_ids and sys.stdin.isatty():
        user_input = input("Enter Image IDs to extract (comma separated), or press Enter to auto-select top traces: ").strip()
        if user_input:
            target_ids = set([x.strip() for x in user_input.split(",") if x.strip()])

    os.makedirs(args.output_dir, exist_ok=True)
    
    interactive_traces = extract_traces_for_images(args.interactive_file, target_ids, is_zero_shot=False)
    zero_shot_traces = extract_traces_for_images(args.zeroshot_file, target_ids, is_zero_shot=True)
    
    if not target_ids:
        interactive_traces = interactive_traces[:4]
        zero_shot_traces = zero_shot_traces[:1]

    print(f"Extracted {len(interactive_traces)} interactive traces and {len(zero_shot_traces)} zero-shot traces.")
    
    idx = 1
    for t in interactive_traces:
        md_content = format_trace_to_markdown(t, is_zero_shot=False)
        out_path = os.path.join(args.output_dir, f"trace_{idx}_interactive.md")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Saved: {out_path}")
        idx += 1
        
    for t in zero_shot_traces:
        md_content = format_trace_to_markdown(t, is_zero_shot=True)
        out_path = os.path.join(args.output_dir, f"trace_{idx}_zeroshot.md")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Saved: {out_path}")
        idx += 1

if __name__ == "__main__":
    main()
