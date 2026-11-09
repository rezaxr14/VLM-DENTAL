# Results in separate folders: flatten, aggregate, publish

`NotebookEvaluations` saves every cohort in its own run folder, and uploads each one to the results repository as its own bundle
(`evaluations/...`). `scripts/aggregate_eval.py` and `scripts/aggregate_k_sweep.py` read one flat folder of `eval_*.jsonl` files. A GRPO policy
trained with the rules/skills harness writes the same file name as the one trained without it, so the two cannot share a folder. This step
connects the two without moving, editing or re-uploading any saved cohort.

## Folders

`scripts/flatten_results.py` copies results into `data/evaluations_flat/` (ignored by git), one folder per area. The Final_Evaluation cell uses one such
root per K (`data/evaluations_flat_k4`, `data/evaluations_flat_k8`), so two chosen K never overwrite each other:

| Folder | Content |
|---|---|
| `test/`, `dev/` | baseline, SFT and GRPO trained without the harness, at the chosen K; `dev/` also holds the tool and grounding probes |
| `test__harness_<mode>/`, `dev__harness_<mode>/` | GRPO trained with that harness at the chosen K, plus the baseline and SFT controls (harness independent) |
| `k_sweep/k<K>/`, `k_sweep__harness_<mode>/k<K>/` | the unablated dev result of every K |
| `rules_skills/<cohort>/`, `speculative/<cohort>/` | `results.json`, `summary.json`, `identity.json`, `runtime.json` of each cohort |

Every area has a `copied_files.json`: the list of what was copied into it (source path, sha256, rows, profile tag, stage, K). A later run removes only the files an earlier
list named, so a changed K never leaves stale files. Other profile tags and stages are never read. Two run folders that map to one flat
name stop the script before anything is copied.

## Commands (also available as notebook cells: GRPO section 12, Final_Evaluation section 9)

```bash
python scripts/flatten_results.py --k <chosen K>
python scripts/aggregate_eval.py --output-dir data/evaluations_flat/test --dataset dentex --split test        # each test/dev area
python scripts/aggregate_k_sweep.py --sweep-dir data/evaluations_flat/k_sweep --reference-dir data/evaluations_flat/dev \
    --log-template "data/models/<grpo run name with {k}>/grpo_training_log.jsonl"
python scripts/publish_aggregates.py --areas test dev --k <chosen K> --tag <profile tag>
```

Run the commands once per K (the Final_Evaluation cell does this with one root folder per K).

## Hub layout

`scripts/publish_aggregates.py` uploads each area as separate files, never into the `evaluations/` tree:

```
aggregates/dentex/<area>/<profile tag>/<stage>/<k<K> | sweep>/run_<id>/<path inside the area>
```

`run_<id>` hashes the area, tag, stage, K and the sha256 of every file. The same files upload once; a changed table goes to a new folder; nothing
already on the Hub is overwritten. The paired harness-on versus harness-off summaries written by `publish_paired_summaries` stay under
`evaluations/summaries/`. The results repository must already exist and `HF_TOKEN` needs write access. `--dry-run` lists the files and needs neither.
