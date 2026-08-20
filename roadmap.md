# VLM-DENTAL: Agentic Radiologist Project Roadmap

This document outlines the current state of the VLM-DENTAL project, the milestones we've achieved, and the remaining steps required to deploy the final autonomous dental diagnostic agent.

---

## 🟢 Completed Milestones (What We've Built)

### 1. Data Pipeline & Environment
- **Dataset Consolidation:** DENTEX loading, preprocessing, and splitting pipeline built and tested (`dental_agent/data/dentex.py`). Tufts Dental Database and other additional datasets are planned (see Left To Do) but not yet integrated — no loader exists for them yet.
- **Colab/Kaggle Architecture:** Set up modular, memory-efficient notebooks (`VLM_Dental_Colab_TraceGen.ipynb`, `VLM_Dental_Colab_YOLO.ipynb`, `VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb`). Implemented smart storage routing to use ephemeral disk space for heavy datasets while safely persisting output traces and weights directly to Google Drive.
- **API Key Management:** Built `api_pool.py` with strict rate pacing, daily limits, and fail-fast exhaustion (no rotating pools).

### 2. Autonomous Trace Generation (Phase 1)
- **Interactive Teacher Loop:** Created an agentic loop where a powerful teacher VLM sequentially invokes tools (zoom, contrast, denoise) to hunt for pathologies, mimicking a real radiologist.
- **Cross-Family Verification:** Implemented a strict verifier (a different model family than the generator) that rejects hallucinated reasoning traces that aren't strictly supported by visual evidence.
- **Bulletproof Parsing Engine:** Overhauled the JSON extractor to intelligently parse mixed XML/JSON outputs, seamlessly repair truncated API responses, and dynamically scavenge broken outputs to keep trajectory loops alive.
- **Self-Correcting Grounding (`nudge_crop`):** Added an 8th tool letting the agent shift/rescale a bounding box it was already given, instead of just trusting `locate_tooth`'s output outright. Trace-gen now shows a real (hint-assisted) detection most of the time, plus a tiered synthetic perturbation — small (25%) or large (30%) offset applied to what's *shown*, never to what's logged internally — so the model has to genuinely look and decide whether to accept or correct, not follow a fixed rule. Tier probabilities and offset ranges are fixed/configured rather than sourced from the real detector's current accuracy, specifically so today's traces don't go stale as the grounding tool improves. Full numbers in `TRACE_GEN_CONFIG.md`.
- **Shared Tool Dispatch (`dental_agent/agent/tool_dispatch.py`):** Trace-gen's LangGraph loop and GRPO's rollout loop (`loop.py`) previously each carried their own copy of "which tools need the image, and which image" — this had already silently diverged (GRPO was compounding crops turn-over-turn against `current_image` instead of `base_image` like trace-gen). Both loops now call one shared `execute_tool_call()`, so this class of bug can't recur when a tool is added.
- **Tool Parameter Control:** Tools previously exposed only a choice of *which* tool to call, not *how much* effect to apply. `denoise` now takes a continuous `strength` (0.0-1.0); `window_level` accepts `center`/`width` overrides on top of its presets; `zoom_crop`'s existing `padding_frac` is now actually exposed in its schema; `contralateral_compare`'s `quadrant` argument (previously accepted but silently unused) now constrains the mirror search to the correct jaw half; `enhance_contrast` (already had a well-designed `factor` parameter) was wired into the registry after being built but never actually registered.

### 3. SFT Training Pipeline (Phase 3)
- **Multi-Modal Collator:** Built `QwenVLDataCollator` to natively parse complex, multi-turn trajectories with dynamically generated image crops directly into Qwen-VL's processor.
- **4-Bit QLoRA Optimization:** Enabled high-efficiency LoRA training for 3B+ parameter models on consumer GPUs (e.g., Colab T4).

### 4. RL/GRPO Implementation (Phase 5)
- **Dual-Adapter Memory Architecture:** Engineered a highly efficient PEFT setup that loads the SFT weights as a frozen `"reference"` adapter and creates a trainable `"grpo_policy"` adapter. By rapidly toggling between them in memory, we compute KL-Divergence penalties without needing a second 3B model loaded into VRAM.
- **VRAM Protections:** Integrated strict cache-clearing mechanisms at the rollout-step level to prevent OOM crashes during heavy multi-turn trajectory sampling.

---

## 🟡 Currently In Progress (User Action Required)

- **Dataset Trace Generation:** Running `scripts/run_trace_gen.py` on Colab/Kaggle to build the synthetic dataset of expert demonstrations — driven by a locally-hosted Qwen/Qwen3.5-9B via a real LangGraph tool-execution loop, now including self-correcting grounding (see above). Only 108 traces exist so far (pre-dating the real-tool-execution rewrite) — this is the actual bottleneck right now, not the training code, which is built out and tested ahead of having data to run it on.
- **YOLO Grounding Tool Training:** Done for now. `yolov8m.pt` trained with 5-fold cross-validation; validation mAP50 ≈ 0.5901 (R ≈ 0.888, P ≈ 0.5457) — past the quality bar, `locate_tooth` is live in the agent loop. Plan is to eventually retrain/generalize this across multiple datasets rather than DENTEX alone (see Left To Do) — trace-gen's perturbation mechanism is deliberately decoupled from this tool's specific accuracy so existing traces won't need regenerating when that happens.

---

## 🔴 Left To Do (Future Milestones)

### Next patch: Dynamic Tool-Call Budget + Reward Retuning
`run_trace_gen`'s `max_tool_calls` already defaults to 50 (some images genuinely need far more tool calls than others). GRPO's own `max_tool_calls` (`grpo.py`) still defaults to 4, and `R_efficiency`'s composite-reward assumption (`dental_agent/rewards/composite.py`) is tuned for a shallow, roughly-fixed tool budget — both need reworking so a longer, genuinely-needed investigation (including nudge_crop corrections) isn't penalized the same as aimless tool spam.

### Dataset Expansion
Add datasets one at a time (starting with Tufts Dental Database), generating traces per dataset and pushing processed data to HuggingFace for easier reuse. Longer-term: generalize `locate_tooth` to work across datasets rather than DENTEX-only.

### Proposal Positioning
`dentex-agentic-vlm-proposal.md` still needs its §3.5/§3.7/§9 novelty claims rewritten to account for OralGPT-Plus (CVPR 2026, code now public) and OralAgent (April 2026) — neither actually threatens the RL-trained-tool-use + DENTEX-native-leaderboard-evaluation core claim, but both need to be cited, not omitted.

### Phase 3: Execute Supervised Fine-Tuning
Once trace generation has real volume, run `VLM_Dental_Colab_SFT.ipynb` to teach the base Qwen-VL model how to use tools and reason like the Teacher VLM.

### Phase 4: Baseline Agent Evaluation
- Create `scripts/run_eval.py` to test the SFT model.
- Compare the model's accuracy in a standard "Zero-Shot" setting vs. an "Agentic Tool-Use" setting to quantify the performance gains of the interactive tools.

### Phase 5: Execute GRPO Reinforcement Learning
Run `VLM_Dental_Colab_GRPO.ipynb` to apply Group Relative Policy Optimization. This will penalize the agent for hallucinating, reward it for accurate diagnoses, and optimize its tool-usage efficiency — including learning when nudge_crop corrections are actually worth the extra call.

### Phase 6: Interactive Clinical UI
- Build a web interface (using Gradio or Streamlit).
- Allow dentists to upload panoramic X-rays and watch the VLM-DENTAL agent visually zoom, enhance, and reason through the image step-by-step in real-time.
