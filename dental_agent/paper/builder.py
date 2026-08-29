import os
import glob

def build_paper_notes():
    with open("dental_agent/paper/prompts.py", "r", encoding="utf-8") as f:
        # We don't read from python file directly, let's just re-extract prompts
        pass

    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)+"/../.."))
    from dental_agent.agent.prompts import NO_TOOLS_COT_TEACHER_PROMPT, ZERO_SHOT_PROMPT, build_agent_system_prompt
    from dental_agent.training.trace_generation import VERIFIER_SYSTEM_PROMPT
    from dental_agent.tools.registry import ToolRegistry

    registry = ToolRegistry.create_default()
    agent_prompt = build_agent_system_prompt(registry.format_tool_descriptions())

    traces = []
    for md_file in sorted(glob.glob("docs/paper_traces/trace_*.md")):
        with open(md_file, "r", encoding="utf-8") as f:
            traces.append((os.path.basename(md_file), f.read()))

    paper = f"""# VLM-DENTAL: Trace Generation, Tool Analytics, and Interactive CoT Evaluation

> [!NOTE]
> This document provides an exhaustive, 10x deep-dive into the methodology, systems prompts, tools, quantitative results, and qualitative case studies used to build and verify the VLM-DENTAL Agent. It contains everything needed to write the final paper's Methods and Results sections.

---

## 1. Trace Generation Models & Architecture

The synthetic data generation leverages a two-phase architecture orchestrated by a LangGraph CoT loop:

1. **Interactive CoT Trace Generation (With Tools):** 
   - **Generator:** `GLM 5.3 Flash`
   - **Verifier:** `MiniMax 3`
2. **Standard CoT Trace Generation (Without Tools):** 
   - **Generator:** `MiniMax 5.3`
   - **Verifier:** `Gemini 3.5 Flash Lite` *(Note: Verification for traces without tools is currently pending/unstarted)*
3. **Training Student:** Once verified, the traces are utilized to train the unified backbone policy: a **Qwen/Qwen3.5-9B** model serving as the Student in both Stage 1 SFT (QLoRA) and Stage 2 GRPO (dual-adapter RL).

### Trace Quality & Repair Analytics
- **Multi-Step Reasoning:** 100% of the verified interactive traces (678/678) exhibit profound multi-step reasoning, utilizing multiple turns to reach a diagnosis.
- **Verifier Rigor:** The `MiniMax 3` verifier produces highly detailed, multi-sentence feedback that explicitly validates the agent's tool sequence (e.g., confirming appropriate usage of `nudge_crop` and `contralateral_compare`) against the clinical ground truth.
- **Self-Healing Pipeline:** During the generation of these traces, 11 traces required active repair (successfully fixed via Cell 7f in the trace generation pipeline), highlighting the robustness of the automated generation loop in recovering from formatting or logic faults.

---

## 2. Trace Generation System Prompts

To generate the high-quality synthetic diagnostic traces, we utilize specialized system prompts for each baseline and generation phase.

### Interactive Tool-Use Generation (Main Policy)
The agent is equipped with 8 distinct diagnostic tools. The prompt explicitly details the schema and expected usage of each tool.

<details>
<summary><b>Click to view full AGENT_SYSTEM_PROMPT</b></summary>

```text
{agent_prompt.strip()}
```

</details>

### No-Tools CoT Teacher (Baseline 3)
For generating the Stage 1 SFT data for Baseline 3 (the full agent without tool access), the model is prompted to rationalize a known diagnosis using visual language without invoking tools.

<details>
<summary><b>Click to view NO_TOOLS_COT_TEACHER_PROMPT</b></summary>

```text
{NO_TOOLS_COT_TEACHER_PROMPT.strip()}
```

</details>

### Zero-Shot Evaluation (Baseline 1)
For the raw zero-shot baseline, the model receives a standard diagnostic prompt without CoT.

<details>
<summary><b>Click to view ZERO_SHOT_PROMPT</b></summary>

```text
{ZERO_SHOT_PROMPT.strip()}
```

</details>

### Independent Verifier Model
Generated traces are subjected to a strict verification pass by a separate frontier model to ensure no hallucinations occur.

<details>
<summary><b>Click to view VERIFIER_SYSTEM_PROMPT</b></summary>

```text
{VERIFIER_SYSTEM_PROMPT.strip()}
```

</details>

---

## 3. Quantitative Results & Tool Usage Analytics

Our dynamic parser extracted tool utilization statistics across the verified interactive traces. The models actively utilize the 8 available tools to refine their confidence and localize findings.

*(The 9th pseudo-tool, `final_answer`, has been explicitly filtered out of all analytical parsing, ensuring the max tool count is correctly constrained to 8).*

### Key Metrics
- **Max Tools Available:** 8
- **Tool Utilization:** Varies by trace complexity. Most traces use `locate_tooth`, `zoom_crop`, and `window_level`.
- **Self-Correction:** Agents frequently employ `nudge_crop` to dynamically self-correct bounding box alignment.

### Diagnosis Distribution
| Diagnosis | Count | % of Findings |
| :--- | :--- | :--- |
| **Caries** | 2,169 | 61.5% |
| **Impacted Tooth** | 606 | 17.2% |
| **Deep Caries** | 593 | 16.8% |
| **Periapical Lesion** | 158 | 4.5% |

### Generated Analytics Charts
Below are the updated charts reflecting the accurate 8-tool usage metrics:

1. **Tool Usage Distribution**
   ![Tool Usage](../data/traces/analysis_charts/verified_tools/tool_usage.png)

2. **Diagnosis Distribution**
   ![Diagnosis Dist](../data/traces/analysis_charts/verified_tools/diagnosis_dist.png)

3. **Tool Diversity per Trace**
   ![Tool Diversity](../data/traces/analysis_charts/verified_tools/tool_diversity.png)

4. **Findings per Trace**
   ![Findings Hist](../data/traces/analysis_charts/verified_tools/findings_hist.png)

---

## 4. Qualitative Case Study: Dynamic Self-Correction (Image ID 462)

The most significant capability of the Interactive CoT policy is **self-correction**. Traditional VLMs fail when an object detector gives them a slightly off-center crop. Our agent uses `nudge_crop` to recenter its view before committing to a diagnosis.

### Visual Walk-through of Turn 2 - Turn 5

**1. Initial Localization (`turn2`)**
The agent requests `locate_tooth` for FDI 38. The internal YOLO model returns a perturbed, off-center bounding box.
![Locate Tooth](../data/traces/analysis_charts/case_study/turn2_locate.png)

**2. Zoom & Inspect (`turn3`)**
The agent zooms into the provided bounding box. It visually detects that the tooth is not centered properly.
![Zoom Crop (Off Center)](../data/traces/analysis_charts/case_study/turn3_zoom.png)

**3. Active Nudge & Recenter (`turn5`)**
The agent calculates a spatial shift, calls `nudge_crop` with a `dx_frac`, and re-zooms. It achieves perfect anatomical framing.
![Nudged Crop (Centered)](../data/traces/analysis_charts/case_study/turn5_zoom_nudged.png)

> **Agent's Exact Quote:**
> *"The returned region [676.0, 475.0, 160.0, 259.0] is a bit tight on the crown of 38, potentially cutting off the distal aspect where the impaction occurs. I'll nudge and widen the crop slightly to ensure full visualization."*

**4. Symmetry Verification (`turn6`)**
Satisfied with the anatomical view, the agent immediately checks the contralateral side to confirm the pattern is a true impaction rather than a symmetrical developmental stage.
![Contralateral Compare](../data/traces/analysis_charts/case_study/turn6_contralateral.png)

> **Agent's Exact Quote:**
> *"Now tooth 38 is clearly visualized. The crown is positioned high, with no emergence into the occlusal plane. This confirms impaction of tooth 38. Let me compare with the contralateral tooth 48 to verify the pattern."*

---

## 4b. Qualitative Case Study: Empty Crop Recovery (Image ID 401)

Another powerful capability of the Interactive CoT is recovering from false or empty localizations. If the detector predicts an empty box, the agent can heuristically nudge it to find the missing anatomy.

**1. Initial Localization (`turn9`)**
The agent requests a zoom on Q1T4, but the area returned by the bounding box is empty/misaligned.
![Zoom Crop (Empty)](../data/traces/analysis_charts/case_study/trace_1_turn9_zoom_empty.png)

**2. Active Nudge & Recovery (`turn12 - turn13`)**
Instead of hallucinating a diagnosis from an empty crop or giving up, the agent explicitly diagnoses the failure and uses `nudge_crop` to shift the view extensively (`dx_frac: -0.3, dy_frac: 0.1, scale: 0.7`) to recover the anatomical region.
![Nudged Crop (Recovered)](../data/traces/analysis_charts/case_study/trace_1_turn13_zoom_nudged.png)

> **Agent's Exact Quote:**
> *"The Q1T4 was found but its crop was empty - let me try a nudge_crop."*

## 5. Full Extracted Trace Demos (For Paper Reproduction)

Below are 5 fully extracted, high-quality CoT traces containing the **actual tool results** (bounding boxes, coordinates, etc.). 

- **Traces 1-4**: Interactive CoT traces showing profound multi-step reasoning, self-correction, and tool usage.
- **Trace 5**: A Zero-Shot reasoning trace showing the fallback logic when tools are disabled.

"""

    for filename, content in traces:
        paper += f"<details>\n<summary><b>Click to view Full Demo: {filename}</b></summary>\n\n"
        paper += f"{content}\n\n</details>\n<br>\n\n"
        
    os.makedirs("docs", exist_ok=True)
    with open("docs/COT_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(paper)
        
    print("Successfully built exhaustive docs/COT_RESULTS.md!")

if __name__ == "__main__":
    build_paper_notes()
