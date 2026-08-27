import json
import sys

filename = 'data/traces/train_cot_traces_unverified.jsonl'
try:
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        
    traces = []
    for line in lines:
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    unique_ids = set()
    for t in traces:
        if 'image_id' in t:
            unique_ids.add(t['image_id'])
            
    print(f'Total Traces: {len(traces)}')
    print(f'Unique Image IDs: {len(unique_ids)}')
except Exception as e:
    print(f'Error: {e}')
