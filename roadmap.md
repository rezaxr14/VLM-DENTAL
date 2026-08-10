# VLM-DENTAL: Agentic Radiologist Project Roadmap

This document outlines the current state of the VLM-DENTAL project, the milestones we've achieved, and the remaining steps required to deploy the final autonomous dental diagnostic agent.

---

## 🟢 Completed Milestones (What We've Built)

### 1. Data Pipeline & Environment
- **Dataset Consolidation:** Successfully merged and formatted the DENTEX and Tufts Dental databases.
- **Colab/Kaggle Architecture:** Set up modular, memory-efficient notebooks (`VLM_Dental_Colab_Master`, `SFT`, `GRPO`). Implemented smart storage routing to use ephemeral disk space for heavy datasets while safely persisting output traces and weights directly to Google Drive.
- **API Key Management:** Built `api_pool.py` with robust round-robin rotation, automatic rate-limit handling, and fallback logic for Gemini and Anthropic APIs.

### 2. Autonomous Trace Generation (Phase 1)
- **Interactive Teacher Loop:** Created an agentic loop where a powerful teacher VLM sequentially invokes tools (zoom, contrast, denoise) to hunt for pathologies, mimicking a real radiologist.
- **Cross-Family Verification:** Implemented a strict verifier (e.g., Claude 3.5 Sonnet) that rejects hallucinated reasoning traces that aren't strictly supported by visual evidence.
- **Bulletproof Parsing Engine:** Overhauled the JSON extractor to intelligently parse mixed XML/JSON outputs, seamlessly repair truncated API responses, and dynamically scavenge broken outputs to keep trajectory loops alive.

### 3. SFT Training Pipeline (Phase 3)
- **Multi-Modal Collator:** Built `QwenVLDataCollator` to natively parse complex, multi-turn trajectories with dynamically generated image crops directly into Qwen-VL's processor.
- **4-Bit QLoRA Optimization:** Enabled high-efficiency LoRA training for 3B+ parameter models on consumer GPUs (e.g., Colab T4).

### 4. RL/GRPO Implementation (Phase 5)
- **Dual-Adapter Memory Architecture:** Engineered a highly efficient PEFT setup that loads the SFT weights as a frozen `"reference"` adapter and creates a trainable `"grpo_policy"` adapter. By rapidly toggling between them in memory, we compute KL-Divergence penalties without needing a second 3B model loaded into VRAM.
- **VRAM Protections:** Integrated strict cache-clearing mechanisms at the rollout-step level to prevent OOM crashes during heavy multi-turn trajectory sampling.

---

## 🟡 Currently In Progress (User Action Required)

- **Dataset Trace Generation:** Running `run_daily_trace_generator.py` on Colab to build the synthetic dataset of expert demonstrations.
- **YOLO Grounding Tool Training:** Running the `yolov8m.pt` training loop for 500 epochs to create the dense bounding-box tool for the VLM agent.

---

## 🔴 Left To Do (Future Milestones)

### Phase 3: Execute Supervised Fine-Tuning
Once trace generation is complete, run `VLM_Dental_Colab_SFT.ipynb` to teach the base Qwen-VL model how to use tools and reason like the Teacher VLM.

### Phase 4: Baseline Agent Evaluation
- Create `scripts/run_eval.py` to test the SFT model.
- Compare the model's accuracy in a standard "Zero-Shot" setting vs. an "Agentic Tool-Use" setting to quantify the performance gains of the interactive tools.

### Phase 5: Execute GRPO Reinforcement Learning
Run `VLM_Dental_Colab_GRPO.ipynb` to apply Group Relative Policy Optimization. This will penalize the agent for hallucinating, reward it for accurate diagnoses, and optimize its tool-usage efficiency.

### Phase 6: Interactive Clinical UI
- Build a web interface (using Gradio or Streamlit).
- Allow dentists to upload panoramic X-rays and watch the VLM-DENTAL agent visually zoom, enhance, and reason through the image step-by-step in real-time.
