# Inference-time skills (Phase 2)

## Design and defaults

Chosen design (a): a `read_skill` harness action in the **shared dispatcher**, outside the unchanged eight-tool registry. The normal loop rejects it when the harness is disabled. No trace-generation or GRPO training entry point enables this feature. Existing trace schemas and to_dict output are unchanged with flags OFF. Enabled episodes add optional `harness_events` with listed/loaded names, versions, file digests, turn index and elapsed time. Skill loads do not spend the diagnostic-call budget, but are bounded by the existing assistant-turn cap. Full injection records its loads at turn zero.

`HarnessOptions(rules_from_file=False, skills=False, skills_mode='off')` is the default. `index_only` means index plus on-demand loading; `full_inject` means index plus every body. A non-off mode requires `skills=True`. The prompt contains no skill bodies in index mode. The five initial files use JSON front matter (a YAML subset), explicit source paths, versions and inspectable body text. No new clinical thresholds or test-set-derived findings were added. Every individual claim and source is in `SKILL_CLAIMS.json`.

## Execution

Run `notebooks/VLM_Dental_Colab_Rules_Skills.ipynb` after training in a fresh Kaggle/Colab session. It contains isolated checkout, optional dependency setup, credentials, four hardware profiles, tagged SFT/GRPO adapter selection, fixed-dev provenance checks, the paired run, summary/events, amendment inspection and export. It never overwrites the training environment or kills any process. Install the patches in the source repository first; the notebook cannot fetch an unpublished branch.

```
python scripts/harness/run_dev.py --model-id /path/to/base --adapter-path /path/to/adapter \
  --precision qlora --qlora-compute-dtype fp16 --dev-split data/splits/dentex_dev.json \
  --conditions baseline rules_only rules_skills_index rules_skills_full --limit 2 \
  --output paper/generated/harness_dev_smoke.json
```

The runner reads the **train-derived dev split only**, checks provenance, refuses missing images and existing outputs, and keeps paired IDs fixed. It uses the existing agent loop, parser, dispatcher, geometry, ground-truth extraction and set-level matcher. Pre-registered metrics: pooled exact F1 and invented-findings rate = unmatched predictions/all predictions (null when denominator is zero). Paired image bootstrap differences accompany format-failure counts. This is an experimental dev runner; the frozen matrix and its six conditions remain unchanged.

`proposals/frozen_protocol_amendment.patch` is an unapplied proposal. Nothing edits the frozen experiment section automatically. Results are pending a real checkpoint/hardware run. Scripted synthetic tests establish plumbing and OFF equivalence only, not model quality or clinical benefit.
