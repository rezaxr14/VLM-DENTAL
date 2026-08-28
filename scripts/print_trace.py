import json

def get_trace():
    with open('data/traces/train_cot_traces.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                t = json.loads(line)
                if t.get('image_id') == 462:  # using 462 because I saw its rich trace earlier
                    # The trace is in the list
                    steps = t.get('trace', t.get('trajectory', []))
                    if isinstance(steps, dict) and 'turns' in steps:
                        pass # unverified format
                    elif isinstance(t, dict):
                        # The verified format puts turns directly as keys? Wait. 
                        # Let's just print the first 2000 chars of the raw line to see it exactly
                        print("RAW LINE 462:")
                        print(line[:2500])
                    break
            except Exception as e:
                pass
get_trace()
