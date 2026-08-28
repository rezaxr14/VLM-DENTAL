import json
from pathlib import Path

traces_dir = Path("data/traces")
for p in traces_dir.glob("*.jsonl*"):
    print(f"\nScanning: {p.name}")
    total = 0
    with_tools = 0
    no_tools = 0
    multi_blob_suspicious = 0
    fake_tool_call_count = 0
    has_hallucinated_xml = 0
    statuses = {}
    bad_indices = []
    
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
                st = rec.get("status", "no_status")
                statuses[st] = statuses.get(st, 0) + 1
                
                traj = rec.get("trajectory", rec)
                t_calls = traj.get("tool_calls", [])
                if len(t_calls) > 0:
                    with_tools += 1
                else:
                    no_tools += 1
                
                raw_str = json.dumps(rec)
                has_bad = False
                if "<fake_tool_call>" in raw_str:
                    fake_tool_call_count += 1
                    has_bad = True
                if "<tool_call>" in raw_str:
                    has_hallucinated_xml += 1
                    has_bad = True
                
                for msg in traj.get("messages", []):
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        # check multi json code blocks or unparsed XML or repeated action
                        if c.count("```json") > 1 or c.count('"thought":') > 2 or c.count('"action":') > 2 or "<tool_call>" in c:
                            multi_blob_suspicious += 1
                            has_bad = True
                            break
                if has_bad:
                    bad_indices.append((idx, rec.get("image_id"), st))
            except Exception as e:
                pass
                
    print(f"Total: {total}, With tools: {with_tools}, No tools: {no_tools}, Statuses: {statuses}")
    print(f"fake_tool_call: {fake_tool_call_count}, <tool_call>: {has_hallucinated_xml}, multi-blob: {multi_blob_suspicious}")
    print(f"Total dirty records: {len(bad_indices)}")
    if bad_indices[:5]:
        print(f"Sample dirty records: {bad_indices[:5]}")
