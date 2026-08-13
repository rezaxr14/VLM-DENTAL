# Visualization Prompts for VLM-DENTAL

---

## 1. High-Level System Architecture
**Prompt:**
> Please generate a highly visual Mermaid graph (flowchart TD) illustrating the high-level architecture of the VLM-DENTAL pipeline. It should show the flow of data from the raw DENTEX dataset into the Trace Generation subsystem, which interacts with a local Vision-Language Model (Qwen3-VL). The generated traces are then sent to a Strict Verifier subsystem (routed across external APIs). Traces that pass verification are collected into the final Synthetic SFT Dataset used for fine-tuning. Make the graph look clean and professional, using subgraphs where appropriate.

---

## 2. The LangGraph Agent Loop (Trace Generation)
**Prompt:**
> Please generate a detailed Mermaid graph (stateDiagram-v2) for the LangGraph-based Trace Generation loop in a medical AI project. 
> The system starts with an `initial_state` containing a base X-ray image and a ground-truth hint.
> It enters the `reasoning_node`, which calls a local vLLM (Qwen3-VL) to get the agent's next action. 
> The output is parsed. If it's a tool call, the state transitions to the `tools_node`, where the tool is executed against a `ToolRegistry` and the visual/text result is appended to the state, before looping back to `reasoning_node`.
> If the parsing fails, it loops back to `reasoning_node` up to 3 times for self-correction.
> If a `final_answer` is given (and at least one tool was used), it routes to the `END` state.
> Please highlight the recursive nature of the loop.

---

## 3. Tool Registry & Execution Pipeline
**Prompt:**
> Please generate a Mermaid graph (flowchart LR) showing how the Tool Execution subsystem works. 
> On the left, we have the `ToolRegistry`. The agent can request one of several tools:
> - `zoom_crop` (crops the image)
> - `window_level` (adjusts contrast for bone/enamel/soft tissue)
> - `denoise` (applies bilateral or median filtering)
> - `contralateral_compare` (crops the opposite side of the jaw for symmetry comparison)
> - `locate_tooth` (runs a YOLOv8m grounding model to find the bounding box of a specific FDI tooth number)
> All image-based tools always act on the original, uncorrupted `base_image` to prevent crop drift. 
> The output of the tools (either a new PIL Image or text) is packed into a multimodal observation and returned to the agent. Please make it visually organized.

---

## 4. API Pool & Verifier Subsystem
**Prompt:**
> Please generate a Mermaid graph (flowchart TD) mapping out the `API Pool & Verifier` subsystem.
> The Batch Runner requests a verification of an agent's reasoning trace. 
> The request hits the `ProviderPool`, which acts as a load balancer and rate limiter. 
> The `ProviderPool` checks its state (daily limits and 5-minute cooldowns) and round-robins the request to the first available external API provider: `NVIDIA NIM`, `Groq`, `OpenRouter`, or `Gemini`. 
> The chosen API runs a Strict Verifier prompt against the candidate trace and the X-ray image. 
> If the trace hallucinates or isn't grounded in the visual evidence, it outputs `{"grounded": false}`. If it passes, it outputs `{"grounded": true}`. 
> The architecture should emphasize the fallback routing and cooldown mechanisms that prevent rate-limit crashes.

---

## 5. SFT (Supervised Fine-Tuning) Pipeline
**Prompt:**
> Please generate a Mermaid graph (flowchart LR) illustrating the SFT (Supervised Fine-Tuning) pipeline for the Vision-Language Model.
> Start with the `Verified Traces Dataset` (the output from the Verifier subsystem).
> Show how these traces are converted into a `Conversational SFT Format` (mapping turns, tool calls, and observations into multi-turn chat messages).
> Then show the `Training Loop` where a base `Qwen3-VL-8B` model is fine-tuned using LoRA/QoRA on these examples to produce the `SFT Dental Agent`.
> Highlight that this process teaches the model *how* to use the simulated radiologist tools effectively.

---

## 6. GRPO (Generative Reward Policy Optimization) + LangGraph Pipeline
**Prompt:**
> Please generate a complex Mermaid graph (flowchart TD) detailing the GRPO (Generative Reward Policy Optimization) Reinforcement Learning pipeline integrated with LangGraph.
> Show the `SFT Dental Agent` acting as the starting policy.
> For each training step, the agent interacts with the `LangGraph Environment` (which provides the image and executes tools via the `ToolRegistry`).
> The agent generates multiple candidate trajectories (rollouts).
> These trajectories are evaluated by a `Reward Function` that checks two things: 1) Format correctness (valid JSON tool calls) and 2) Diagnostic accuracy (does the final answer match the ground truth?).
> The rewards are used to compute advantages and update the model weights via GRPO, resulting in the `Final RL-Tuned Dental Agent`.
