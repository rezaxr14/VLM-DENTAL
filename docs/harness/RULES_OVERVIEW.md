# Harness rules contract v1.1.0

This extraction records the existing prompt and enforcement behavior verbatim. No test-set records, images, metrics or annotations were used to derive this contract.

`dental_agent/harness/rules.md` has YAML-compatible JSON front matter: version, numbered rules, provenance, enforcement bindings and the verbatim system prompt template. The eight numbered agent guidelines become R001–R008. R009–R012 bind the four existing reward components. R013 binds the deterministic healthy-case rejection; R014–R017 record the independent verifier's four instruction clauses. R018 records orientation; R019–R024 inventory the optional judge, legacy reward functions and composite aggregator without activating or rewriting them. R025 binds the existing default reward-weight configuration.

The prompt builder accepts `harness=HarnessConfig(rules_from_file=True)`; the default is OFF. Existing call sites pass no harness and retain their prior code path. Enabled rendering substitutes only tool definitions and dataset class strings, preserving JSON braces, whitespace, examples and taxonomy. It validates implementation bindings before rendering. This phase does not enable a harness through training YAML or modify running scripts.

Bindings validate existing functions by AST digest and validate the verifier instruction text separately. This detects implementation drift without duplicating reward math or changing outcomes. The verifier remains an independent LLM: instruction binding is not a guarantee of a particular clinical judgment. The deterministic rejection gate is tested separately. No new enforcement is added.

## Discrepancies retained

- `training/rewards.py` contains legacy reward definitions, including `<think>` formatting and different single-finding scoring. They conflict with the active JSON-only prompt/core rewards. They are inventoried and digest-bound, not silently unified or enabled.
- The optional external `reward_judge` is distinct from the trace-generation verifier; its instructions and outputs are not equivalent to deterministic reward terms. Its implementation is bound without introducing API calls in tests.

- Prompt: final answer must be a list. Format reward: a dict is accepted too; a list's members are not schema-validated by that reward.
- Prompt: varied use of eight tools. Efficiency reward: discounts locate/nudge, penalizes some repetition and excess budget; it does not enforce tool diversity.
- Prompt: all findings investigated with tools. Accuracy reward compares final findings; it does not independently validate visual evidence or prove grounding.
- Verifier explicitly rejects fabricated outputs/XML and incorrect healthy-case findings; those gates are not all expressed as numbered agent guidelines.
- Prompt examples, including their clinical conclusions and healthy-survey sequence, are preserved verbatim so the golden prompt remains unchanged.
- The runtime no-tools, teacher no-tools and raw zero-shot prompts are distinct and remain outside this first with-tools contract. They must not be silently unified.
- `dental_agent/paper/prompts.py` writes scratch prompt exports; it is not the active prompt definition.

## Reproduce

```
python -m pytest tests/test_harness_rules.py -q
python -m pytest -q
python scripts/paper/verify_paper.py
```

Golden tests cover default/OFF/ON for DENTEX, Tufts, Tufts-all and custom taxonomy, including marker-like literal tool text. Regression fixtures are synthetic, contain no dataset image IDs, and exercise existing reward functions and the exact existing healthy-case gate. They do not establish clinical validity or substitute for an archived real-trace verifier audit. Frozen experiment protocol and all reward/verifier source files remain unchanged.
