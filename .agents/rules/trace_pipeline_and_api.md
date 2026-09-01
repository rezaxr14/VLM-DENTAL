---
trigger: always_on
---

# Trace Generation, Verification & API Discipline Rules

## 1. Trace File Naming Convention (CRITICAL)
- **Rule:** `data/traces/train_cot_traces.jsonl` is the CANONICAL verified trace file used by all downstream pipelines (SFT, GRPO, YOLO notebooks).
- Legacy traces are preserved at `train_cot_traces.jsonl.old`.
- Raw unverified traces are written to `data/traces/train_cot_traces_unverified.jsonl`.
- Verification promotes passing traces to `train_cot_traces.jsonl`.
- **DO NOT** rename or create alternative output filenames (e.g. `cot_traces_aim1.jsonl`).

## 2. Decoupled Generation & Verification Pipeline
- **Rule:** Generation and verification are independent phases running at different speeds.
  - **Generation**: Writes raw traces to `_unverified.jsonl` as fast as the generator allows.
  - **Verification**: Reads `_unverified.jsonl`, verifies via external APIs (`ProviderPool`), and promotes passing traces.
- **Rule:** Both phases must support resume (tracking processed image IDs).
- **Rule:** When `GENERATOR_PROVIDER=local`, generation must NOT be stalled by verifier rate limits.

## 3. No Retries on API Errors, Except Explicit 429 Opt-In (CRITICAL)
- **Rule:** By default, on 429 or any API error, DO NOT retry. Stop immediately and exit.
- **Exception:** `IGNORE_429=true` in the environment opts into up to 10 retries specifically for 429 errors (5s apart).

## 4. No Real API Calls for Testing (CRITICAL)
- **Rule:** NEVER waste API budget for testing or verification. Always use mock API calls, fake stubs, or local inference.
