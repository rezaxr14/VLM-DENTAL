"""select_longest_traces.py: picks the longest traces and the probe file is filtered correctly by the dataset."""

import json
import subprocess
import sys
from pathlib import Path

from dental_agent.training.sft import DentalSFTDataset

ROOT = Path(__file__).resolve().parent.parent


class _Proc:
    class tokenizer:
        pad_token_id = 0
        eos_token_id = 1


def test_probe_selects_longest_and_dataset_filters_it_by_ceiling(tmp_path):
    lengths = {1: 5000, 2: 15000, 3: 9000, 4: 15500}
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps({"image_id": i, "dataset": "dentex", "messages": []}) for i in lengths) + "\n")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "canonical_resize": True,
        "lengths_by_file_and_id": {f"traces.jsonl::dentex::{i}": n for i, n in lengths.items()},
    }))
    out = tmp_path / "probe.jsonl"
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "select_longest_traces.py"), "--traces", str(traces), "--manifest",
         str(manifest), "--top", "2", "--output", str(out), "--ceilings", "10000", "16384"],
        capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert [json.loads(l)["image_id"] for l in out.read_text().splitlines()] == [4, 2]
    assert "<=  10000:     2 / 4" in res.stdout and "<=  16384:     4 / 4" in res.stdout

    # The probe file name is not in the manifest; the '<dataset>::<id>' key must still match so --max-seq-len filters it
    manifest.write_text(json.dumps({
        "canonical_resize": True,
        "lengths_by_file_and_id": {f"traces.jsonl::dentex::{i}": n for i, n in lengths.items()},
        "lengths_by_image_id": {f"dentex::{i}": n for i, n in lengths.items()},
    }))
    kept = lambda ceiling: [r["image_id"] for r in DentalSFTDataset(
        out, processor=_Proc(), max_seq_len=ceiling, token_lengths_manifest=manifest, canonical_resize=True).records]
    assert kept(16384) == [4, 2]
    assert kept(15200) == [2]
