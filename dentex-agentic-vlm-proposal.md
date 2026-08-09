# Research Proposal
## An Agentic, Tool-Augmented Vision-Language Model for Panoramic Dental Radiograph Diagnosis

*Companion document to: agentic-orthodontic-vlm-research-landscape.md (literature review, dataset survey, gap analysis)*

---

## Core aim (one paragraph — what this project is)

We are building and training one thing: a Qwen3-VL-based reasoning agent that looks at a panoramic dental X-ray, decides for itself which regions need a closer look, calls a small set of purpose-built tools (zoom, contrast enhancement, tooth segmentation, FDI numbering) to gather evidence, and then commits to a structured finding — quadrant, tooth number, and diagnosis — with a stated confidence. It learns to do this not from imitating labeled examples alone, but from a reward signal (GRPO reinforcement learning) that scores whether its final answer was correct, whether its tool calls were sensible, and whether it stayed within a well-formed output format. Training data and evaluation are both designed to need **no clinician access**: reasoning traces are synthesized and quality-checked by larger frontier LLMs (used as generators and as judges via API), and validation leans on DENTEX's own dentist-verified ground truth rather than a prospective expert study. The paper's single empirical question is: **does giving a VLM tools + RL-trained reasoning measurably beat both a zero-shot VLM and a conventional supervised detector on the same benchmark (DENTEX), and how much of the gain comes from the tools versus the RL?** Everything else in this document exists to answer that one question.

---

## 1. Title (working)

**"Think, Look, Measure: A Tool-Augmented, Reinforcement-Learned Vision-Language Agent for Abnormal Tooth Detection, Enumeration, and Diagnosis on Panoramic Radiographs"**

---

## 2. Abstract (draft, ~160 words)

Vision-language models (VLMs) applied zero-shot to panoramic dental radiographs perform unevenly — strong on obvious findings like implants, weak on subtle pathology like caries and periapical lesions — because they reason over a single static image with no ability to zoom, localize, or verify. We propose an agentic VLM that plans its diagnostic reasoning in natural language (chain-of-thought) and calls a small suite of deterministic and learned tools — region zoom/crop, contrast enhancement, a fine-tuned segmentation/grounding model, and an FDI tooth-numbering function — before committing to a finding. The reasoning policy is fine-tuned with Group Relative Policy Optimization (GRPO) against a composite, automatically verifiable reward, following a recipe shown effective for medical reasoning VQA (MedVLM-R1, Med-R1) and, separately, for general-domain tool-augmented visual reasoning (VTool-R1) — but never combined, and never applied to dentistry. Training data and evaluation are built entirely from public datasets and frontier-LLM-generated/verified synthetic reasoning traces, requiring no prospective clinical data collection. We evaluate on the DENTEX benchmark against zero-shot VLM baselines and prior supervised detectors, with ablations isolating the contribution of tool use and RL independently.

---

## 3. Related Work

### 3.1 Automated cephalometric and panoramic radiograph analysis
The dominant paradigm in dental AI to date is narrow, single-task supervised learning. Cephalometric landmark detection progressed from game-theory and random-forest methods on the original ISBI 2014/2015 challenge sets, through CNN and Faster R-CNN detectors, to attention-augmented and two-stage regression networks evaluated on the more diverse, multi-device Aariz/CEPHA29 benchmark. Reported success-detection-rate figures have improved from roughly 71% to 82% over a decade, and the benchmark's own organizers describe full automation as still unresolved. Panoramic radiograph work follows a parallel trajectory: U-Net-style segmentation, FDI tooth-numbering classifiers, and object detectors (YOLO-family, Faster R-CNN, fine-tuned SAM/MedSAM variants) for caries, periapical lesions, and impactions. The DENTEX benchmark (Hamamci et al., 2023) consolidated this into a single, hierarchically annotated quadrant–enumeration–diagnosis task, and remains the most structured public benchmark for the specific problem this proposal addresses. Across all of this work, the model architecture is a single forward pass trained end-to-end on labels; none of it reasons, plans, or explains itself, and none of it can request a closer look at an ambiguous region.

### 3.2 3D intraoral scans and CBCT
A separate line of work applies geometric deep learning — point-transformer and edge-convolution architectures — to 3D intraoral scans, enabled by the Teeth3DS/Teeth3DS+ benchmarks (3DTeethSeg'22, 3DTeethLand'24) and related CBCT segmentation datasets (ToothFairy2, CTooth). This work is architecturally disjoint from the 2D image-language literature below: no published system reasons over a 3D dental scan in natural language, and mesh/point-cloud encoders do not yet compose naturally with VLM backbones. We treat this as an out-of-scope but closely related direction (see the companion landscape document, §2.3, §5).

### 3.3 Vision-language models applied to dentistry
The application of general-purpose VLMs to dental images is recent and thin. Zero-shot evaluations of GPT-4V/GPT-4o on periapical and panoramic radiographs report strong performance on visually obvious findings (implants, missing teeth) but near-zero sensitivity for subtler pathology such as caries, periapical lesions, and fractures — a pattern consistent with a model reasoning over a single, fixed-resolution image with no mechanism to inspect a suspicious region more closely. A lightly fine-tuned variant ("CGPT-4V") improved mesiodens-detection accuracy substantially over the zero-shot baseline with only modest domain adaptation, suggesting the ceiling on VLM performance in this domain is set more by training/inference design than by the backbone's raw visual capability. One preliminary framework (SLSO) explored a two-stage self-correction loop for generating jaw-cyst findings from panoramic X-rays with a GPT-based VLM — the closest existing approach to an iterative or self-verifying dental VLM pipeline, though it does not call external tools and is explicitly described by its authors as preliminary. No dental paper to date trains (rather than merely prompts) chain-of-thought reasoning, and none integrates callable visual tools.

### 3.4 Reinforcement-learned reasoning in medical vision-language models
Outside dentistry, RL fine-tuning of VLM reasoning has matured quickly since the release of DeepSeek-R1's GRPO recipe. MedVLM-R1 fine-tuned a small (2B-parameter) VLM with GRPO on only ~600 examples and reported a large accuracy jump over the zero-shot baseline across MRI, CT, and X-ray VQA, with an interpretable chain-of-thought as a side effect of the reward design rather than a separate objective. Med-R1 applied the same family of methods across eight imaging modalities on a Qwen2-VL backbone and outperformed a substantially larger zero-shot model, while its own ablations found that simply lengthening the chain-of-thought does not reliably improve accuracy — reward alignment matters more than reasoning length, a finding directly relevant to how we weight our own reward terms (§5.5). Other efforts (Patho-R1, SafeMed-R1) extend this recipe with staged pretraining and adversarial robustness objectives. None of this work has been applied to any dental task, and none of it incorporates callable tools — the reasoning is entirely textual, conditioned on a single static image.

### 3.5 Agentic tool-use in medical and general-domain VLMs
A parallel thread treats the VLM as an orchestrator that calls external specialist tools rather than doing all visual work itself. MMedAgent (EMNLP 2024) trains an LLM "brain" to select among grounding (Grounding DINO), segmentation (MedSAM), classification, and report-generation tools via an instruction-tuned tool-selection dataset. OctoTools proposes a general, extensible agentic tool-use framework not specific to medicine. RadAgents implements a multi-agent, role-specialized workflow for chest-X-ray reading. None of these systems are trained with RL against a verifiable reward; tool selection is learned via supervised instruction-following, not reinforcement.

The system closest in spirit to what we propose is **VTool-R1** (Wu et al., 2025; ICLR 2026), which trains a VLM with GRPO to call Python-based visual-editing tools mid-reasoning, producing genuinely multimodal chains of thought rather than text-only reasoning about a static image. On structured chart/table visual question answering, VTool-R1 (7B) improved accuracy from a 64.7% zero-shot baseline to 71.7% — roughly a 7-point absolute (≈11% relative) gain attributable to learned tool use — and its own ablations similarly found that more tool calls are not monotonically better, echoing Med-R1's finding about reasoning length. VTool-R1 is general-domain (charts and tables), uses generic image-editing primitives, and optimizes a single outcome-accuracy reward with no equivalent of a structured, multi-part clinical label. It has not been applied to any medical or dental task.

### 3.6 Synthetic data generation and LLM-as-judge verification (no clinician required)
A growing body of work shows that frontier LLMs, used via API, can generate and verify reasoning-trace training data at a quality sufficient for downstream fine-tuning, substantially reducing dependence on human (let alone clinical-expert) annotation. Huatuo-o1 used GPT-4o to convert verifiable medical QA pairs into chain-of-thought training data, while MedReason built a knowledge-graph-grounded pipeline for the same purpose, reserving human clinical experts for spot-checking rather than exhaustive verification. Most directly relevant to our design, **OpenMedReason** (2026) constructs reasoning-supervision data for a Qwen3-VL-8B medical VLM — the same backbone family we propose — using a frontier model (GPT-5) purely as a *verifier*: given the image, source context, question, answer, and candidate reasoning trace, the verifier judges only whether the trace is evidentially grounded and rejects traces that assert claims absent from or contradicted by the available evidence, without rewriting the trace itself. The broader LLM-as-judge literature reports that GPT-4-class judges achieve consistency and stability comparable to professional human evaluators on evaluation and preference-ranking tasks, and self-training approaches (e.g., self-improving VLM judges) show this quality can be maintained without any human preference annotation at all, iterating entirely from synthesized preference pairs. This body of work is the direct methodological basis for §5.2 and §6 below: we substitute a frontier-LLM verifier (and DENTEX's own dentist-verified ground truth) for a prospective clinician study.

### 3.7 Positioning of the present work
No published system pairs (a) trained, not merely prompted, chain-of-thought reasoning, (b) callable, clinically-grounded visual tools, and (c) RL-based policy optimization with a structured, graded diagnostic reward, for any dental or broader medical imaging task. VTool-R1 establishes (a)+(c)+tool-calling in a general, non-clinical domain; MedVLM-R1/Med-R1 establish (a)+(c) for medical VQA without tool calling; MMedAgent/OctoTools/RadAgents establish tool orchestration without RL. This proposal is positioned precisely at the intersection these three lines of work have not yet reached, using DENTEX as a ready-made, expert-annotated benchmark, and using frontier-LLM generation/verification (§3.6) in place of a prospective clinical study to keep the entire pipeline buildable without clinician access.

---

## 4. Research objectives (aims)

**Aim 1 — Data.** Construct a chain-of-thought + tool-use instruction dataset grounded in DENTEX, pairing each labeled abnormal tooth with a plausible reasoning trace and the sequence of tool calls that would justify the label, generated and quality-controlled entirely via LLM APIs (no clinician involvement — see §5.2).

**Aim 2 — System.** Design and implement an agentic VLM pipeline: a reasoning core (Qwen3-VL) that iteratively calls a tool suite (zoom/crop, contrast enhancement, segmentation/grounding, FDI numbering) and outputs a structured quadrant–enumeration–diagnosis report.

**Aim 3 — Training.** Fine-tune the reasoning policy in two stages — supervised fine-tuning (SFT) on the Aim 1 dataset for format/tool-call competence, then GRPO reinforcement learning against a composite verifiable reward — and characterize what RL adds beyond SFT alone.

**Aim 4 — Evaluation.** Benchmark against (i) zero-shot general VLMs (GPT-4o, Qwen3-VL), (ii) the same backbone with SFT only (no RL), (iii) prior supervised specialist detectors, and (iv) DENTEX's own dentist-verified ground truth plus an LLM-judge-based reasoning-grounding check, with ablations isolating tool use and RL independently — entirely dataset- and API-based, with no prospective clinician study required.

### 4.5 Hypotheses — why we expect this to outperform prior approaches
Stated as falsifiable hypotheses, each grounded in a specific precedent rather than asserted on faith:

- **H1 (tools beat no-tools).** The tool-augmented agent will outperform the same backbone reasoning over the whole image only, because the dominant failure mode in existing zero-shot dental VLM evaluations is missed subtle pathology — a resolution/attention problem that a zoom/crop tool directly targets. VTool-R1 demonstrates a ~7-point accuracy gain from learned tool use in a non-clinical domain; we expect a comparable or larger gain here because dental pathology (small caries, thin periapical radiolucencies) is more resolution-sensitive than chart/table reading.
- **H2 (RL beats SFT-only).** GRPO fine-tuning will outperform SFT-only imitation of the same trace dataset, because SFT only teaches the *shape* of correct reasoning while RL directly optimizes the graded accuracy reward, including on the partial-credit cases (right quadrant, wrong diagnosis) that SFT has no signal for. MedVLM-R1 and Med-R1 both report large accuracy gains from this exact transition on comparably small VLM backbones.
- **H3 (competitive with, not necessarily beating, specialist detectors).** We hypothesize the full agent will be competitive with a supervised specialist detector on raw diagnosis F1 while providing an interpretable, auditable reasoning trace and graceful handling of ambiguous cases that a closed-set detector cannot explain. We treat this comparison as genuinely open rather than assumed — if the specialist detector wins on raw accuracy, that is a reportable, honest finding, and the interpretability/tool-audit trade-off becomes the paper's argument rather than a raw win.

H1 and H2 are the paper's primary claims and are the ones the evidence above makes us most confident about; H3 is deliberately framed as a fair comparison rather than a guaranteed win, since overclaiming against a well-tuned specialist detector is the fastest way to lose reviewer trust.

---

## 5. Methodology

### 5.1 Dataset
- **Primary**: DENTEX — 1,005 fully labeled panoramic X-rays (quadrant + FDI enumeration + diagnosis: caries, deep caries, periapical lesion, impaction), 705/50/250 train/val/test split, plus 1,571 unlabeled X-rays for optional pretraining, plus partially labeled quadrant-only and quadrant-enumeration sets for curriculum pretraining of the grounding tool. Critically for a no-clinician pipeline: **DENTEX's ground truth is already clinician-verified at the source** — each image is annotated by a final-year dental student and then verified and corrected by one of three expert dentists with over 15 years of experience (Hamamci et al., 2023). This lets DENTEX's test-set labels stand in for a live clinician review rather than merely approximating one.
- **Secondary (optional, for robustness)**: the 1,512-image multi-condition dataset (fillings/cavities/implants/impacted teeth) and UFBA-style datasets, to test cross-dataset generalization of the trained agent without needing new expert annotation.

### 5.2 Reasoning-trace dataset construction (Aim 1, detail — fully API-based, no clinician)
Because no dataset of dental CoT + tool-call traces exists, generate and verify it entirely via LLM APIs:
1. **Generation**: for each labeled abnormal tooth in DENTEX training data, prompt a strong frontier VLM (used only as a **trace generator**, e.g., via a general-purpose model API) to produce a plausible step-by-step rationale *conditioned on the known ground-truth label*, including which region it would zoom into and what visual evidence supports the diagnosis.
2. **Tool-call insertion**: programmatically insert the corresponding tool-call syntax (crop coordinates from the ground-truth bounding box, expected segmentation output) into the trace so it is consistent with what the real tools return at training time.
3. **Automated verification (replaces human/clinician review)**: pass each generated trace, together with the source image and the DENTEX ground-truth label, to a second, independent frontier model acting purely as a **verifier** — following the OpenMedReason (2026) design, where the verifier judges only whether the trace is evidentially grounded in the image and label (not whether it "sounds right") and rejects any trace that asserts a claim absent from or contradicted by the available evidence. Traces are additionally cross-checked by **self-consistency majority voting**: generate k candidate traces per case and retain only traces whose stated conclusion agrees with the majority, discarding outliers.
4. **Bias control**: use a *different* model family for generation than for verification (e.g., generate with one frontier API, verify with another) so the verifier is not simply grading its own outputs — a known failure mode in single-model LLM-as-judge pipelines.
5. This yields the SFT dataset (Aim 3, stage 1) and a pool of format-verifiable examples for RL reward computation, at a scale anchored to MedVLM-R1's precedent (a few hundred verified examples was sufficient there).

### 5.3 System architecture
**Reasoning core (decided, staged to match actual compute access): Qwen3-VL-2B-Instruct for prototyping now on free/laptop-scale compute, promoted to Qwen3-VL-8B-Instruct for the final reported results once RTX 4090 / server access opens up** (see §5.4 for the full compute-tiered plan). This is not an arbitrary pick: it is the exact backbone family used by Med-R1, MedVLM-R1, VTool-R1, and OpenMedReason, which keeps results directly comparable to the strongest existing precedents in both medical RL-VLM work and general RL+tool-use-VLM work, and it already has a working, documented GRPO training recipe (Hugging Face TRL's VLM-GRPO cookbook) to build from rather than starting from scratch. Deliberately not a frontier-scale model, since MedVLM-R1 and Med-R1 both show small RL-tuned models outperforming much larger zero-shot ones. Output format constrained to a `<think>...</think>`/tool-call block/`<answer>...</answer>` schema so format and accuracy rewards are computable automatically.

Tool suite (implement these four for the core paper; anomaly detection and calibration/measurement are explicitly out of scope for this paper — see §5.6):
1. **Zoom/crop tool** — deterministic function, takes coordinates or a natural-language region description, returns a higher-resolution crop.
2. **Contrast/enhancement tool** — deterministic histogram equalization / windowing.
3. **Segmentation & grounding tool** — a Grounding-DINO- or SAM/MedSAM-style model fine-tuned on DENTEX's quadrant-enumeration boxes, returning a mask/box for a requested tooth or region.
4. **FDI numbering tool** — maps a detected tooth's position to its correct quadrant-enumeration label, isolating this as a distinct, separately-measurable failure mode from diagnosis itself (DENTEX's own hierarchical annotation structure supports this split).

### 5.4 Training pipeline
- **Stage 0**: Pretrain/fine-tune the grounding tool on DENTEX's quadrant and quadrant-enumeration subsets (standard supervised detection training, not part of the novel contribution, but a prerequisite).
- **Stage 1 (SFT)**: Fine-tune the VLM reasoning core on the Aim-1 trace dataset so it reliably produces well-formed reasoning + tool calls (teaches the *shape* of the behavior).
- **Stage 2 (RL / GRPO)**: For each training input, sample a group of G rollouts (e.g., G = 8) from the current policy, score each with the composite reward (§5.5), compute the group-normalized advantage `(reward − group mean) / group std` per rollout, and update the policy — no separate value/critic network needed, following the GRPO recipe used in Med-R1 and MedVLM-R1.
  - **Compute-tiered implementation (decided, matched to actual available resources — three tiers, used in sequence)**:
    - **Tier 1 — now: Kaggle, Colab free tier, RTX 4050 laptop.** Kaggle (~30 GPU-hours/week, P100 16GB or T4×2, ~9–12h sessions, supports background execution — start a job and close the tab) is the venue for anything needing sustained unattended GPU time. Colab free tier (T4, 16GB, ~12h session but needs an active tab without Pro+) suits interactive debugging better than long runs. The 4050 laptop (~6GB VRAM, no quota) is for tool development, data-pipeline code, and 4-bit inference-only debugging of the tool-calling loop — not training. Use this tier for: the trace-generation/verification pipeline (§5.2 — needs no GPU at all, API calls only), Stage 0 (grounding/segmentation fine-tuning), and Stage 1 SFT of the **3B** backbone via TRL + QLoRA (4-bit), which fits comfortably in 16GB.
    - **Tier 2 — this semester: occasional RTX 4090 (24GB) lab access.** Use each session for the harder-to-interrupt work: a first real (not smoke-test) GRPO run on the 3B model with a full group size (G≈8), then — once that pipeline is validated — the first 7B QLoRA SFT/GRPO attempts, which the 16GB free tiers can only run in a heavily constrained way.
    - **Tier 3 — next semester: dedicated 4090 + optional high-VRAM server.** Full-scale 7B GRPO training and the complete reward-weight sweep (§5.5); port to **EasyR1** (multi-node veRL fork, the same framework VTool-R1 was trained with) only if the 4090 becomes throughput-limited for the full ablation matrix — don't default to it before that's actually the bottleneck.
    - **Practical notes**: checkpoint the LoRA adapter frequently given free-tier session/quota limits; push checkpoints to the Hugging Face Hub or Drive rather than ephemeral notebook storage; reduce GRPO's group size (e.g., G=4) when prototyping on free tiers and only restore the full G=8 on the 4090.
- **Stage 3**: Full pipeline evaluation on the held-out DENTEX test set (§6).

### 5.5 Reward function (composite, for GRPO)

`R = w_acc · R_accuracy + w_fmt · R_format + w_tool · R_tool_validity + w_eff · R_efficiency [+ w_judge · R_judge]`

- **R_accuracy** (dominant term, e.g. w_acc = 1.0): graded, not binary — full credit if quadrant + FDI enumeration + diagnosis all match DENTEX ground truth; partial credit (e.g. 0.5) if quadrant + enumeration correct but diagnosis wrong; smaller partial credit (e.g. 0.25) if only quadrant correct; 0 otherwise.
- **R_format** (e.g. w_fmt = 0.2): reward for valid `<think>/<answer>` structure and well-formed tool-call syntax, independent of correctness.
- **R_tool_validity** (e.g. w_tool = 0.2): reward for tool calls that execute successfully and whose returned region plausibly overlaps the eventual claimed tooth location.
- **R_efficiency** (e.g. w_eff = 0.1, penalty term): small negative reward per tool call beyond a budget (e.g., 4 calls), to discourage runaway or redundant tool chains.
- **R_judge** (optional, e.g. w_judge = 0.1): an auxiliary reward from the same frontier-LLM-verifier setup used in §5.2, scoring whether the *final* reasoning trace (not just the label) stays grounded in the tool outputs it cites — a fully automated substitute for what a clinician's qualitative reasoning-review would otherwise provide.
- All weights are hyperparameters to sweep; report the sweep as an ablation.

### 5.6 Explicitly out of scope for this paper (future work)
**Decided**: the anomaly/outlier detection tool (a reconstruction-based autoencoder/diffusion model trained only on healthy regions, flagging pathology outside DENTEX's four labeled classes) is **not** part of this paper's core deliverable. It stays a well-defined follow-up project, reusing this paper's reasoning core and tool-orchestration code, and connecting to the broader unsupervised-anomaly-detection gap identified in the landscape document (§5, option 3).

---

## 6. Evaluation plan (fully dataset- and API-based — no clinician required)

**Baselines** (all evaluated on the identical DENTEX test split):
1. Zero-shot GPT-4o / Qwen3-VL, no tools, single pass.
2. SFT-only agent (Stage 1 only, no RL) — isolates the contribution of RL.
3. Full agent without tool access (RL-tuned but reasoning over the whole image only) — isolates the contribution of tools.
4. Full agent (SFT + tools + RL) — the proposed system.
5. Prior supervised specialist detector (e.g., a YOLOrtho-style or DENTEX-challenge-winning detection pipeline) — the "why not just use a detector" comparison reviewers will ask for.

**Validity anchor in place of a live clinician study**: primary correctness is scored against DENTEX's own dentist-verified ground truth (§5.1) — a benchmark whose own validation work (an independent DMFT-scoring study on DENTEX-trained models) found automated-pipeline agreement with dentists statistically comparable to dentists' agreement with each other (ICC ≈ 0.90 vs. inter-dentist ICC ≈ 0.88). This is the paper's clinical-credibility anchor, cited explicitly rather than asserted informally.

**Metrics**:
- Precision/recall/F1 per diagnosis class, balanced accuracy (matches DENTEX's own reporting convention and the GPT-4o dental study's convention, enabling direct comparison).
- FDI enumeration accuracy, reported separately from diagnosis accuracy.
- Tool-use ablation deltas (with vs. without each individual tool — leave-one-tool-out), directly testing H1.
- RL-vs-SFT-only delta, directly testing H2.
- Tool-call efficiency (mean calls per case) and tool-selection accuracy.
- Calibration: expected calibration error between the agent's stated confidence and actual correctness.
- **Reasoning-grounding check (replaces clinician reasoning review)**: the R_judge protocol from §5.5, applied at test time as an evaluation metric rather than only a training reward — reports what fraction of test-set reasoning traces stay evidentially grounded in their own cited tool outputs, using a frontier-LLM judge from a different model family than the one used to generate the traces (§5.2, bias control).
- **Cross-dataset generalization**: zero-shot evaluation on the secondary 1,512-image dataset (different institution/equipment), as an automated stand-in for the "does this generalize beyond one clinic" question a clinician collaborator would otherwise be asked.

**Explicit limitation to state in the paper**: this protocol validates against existing expert-verified labels and automated reasoning-grounding checks, not a prospective, blinded review by a practicing orthodontist on new cases. That is a genuine limitation, not a hidden one — state it plainly in the paper's limitations section and frame a live clinician study as the natural next step for a lab or team with clinical access, rather than pretending the dataset-based validation is equivalent to one.

---

## 7. Timeline (12-month plan)

| Phase | Months | Primary compute | Activities |
|---|---|---|---|
| 1. Setup & data curation | 1–2 | Laptop (4050) + LLM APIs — no GPU needed | Finalize related-work positioning; set up DENTEX pipeline; build and run the generation+verification pipeline (§5.2) to produce the CoT/tool-call trace dataset (Aim 1) |
| 2. Tool suite + Stage 0 | 2–3 | Kaggle free tier (background execution) | Fine-tune grounding/segmentation tool on DENTEX; implement zoom/crop/enhancement (laptop); implement FDI numbering function; unit-test each tool |
| 3. SFT (3B) | 3–4 | Kaggle/Colab free tier (QLoRA, 16GB) | Wire tools into the VLM's tool-calling interface; run Stage 1 SFT on Qwen3-VL-2B; validate format/tool-call reliability |
| 4. GRPO smoke test (3B) | 4–5 | Kaggle/Colab free tier, reduced group size | Get the GRPO loop running end-to-end at small scale (G≈4) to validate reward computation and rollout mechanics before spending scarcer 4090 time |
| 5. First real GRPO run, 3B → 7B | 5–7 | Occasional RTX 4090 (24GB), this semester | Full-group (G≈8) GRPO on 3B; first 7B QLoRA SFT/GRPO attempts once 3B results look sound |
| 6. Full-scale training & reward sweep | 7–9 | Dedicated RTX 4090 / server, next semester | Full 7B GRPO training; complete reward-weight sweep (§5.5); move to EasyR1 only if 4090 throughput becomes the bottleneck |
| 7. Full evaluation | 9–10 | Whichever tier is available (inference-only, cheap) | Run all baselines + ablations; reasoning-grounding and cross-dataset generalization checks (§6) |
| 8. Writing & submission | 10–12 | Laptop | Draft manuscript; internal review; submit; buffer for a first-round revision |

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| No existing CoT/tool-call trace data for dentistry | Fully automated generation + frontier-LLM verification (§5.2); anchor sample size to MedVLM-R1's precedent |
| Compute cost of GRPO | Small backbone (2–7B), LoRA/QLoRA fine-tuning, modest group size (G ≈ 8), capped tool-call budget per rollout |
| Reviewers ask "why not just fine-tune a detector" | Report baseline #5 explicitly; quantify the accuracy/interpretability trade-off honestly, including cases where the detector wins (H3 is a fair test, not a guaranteed win) |
| Hallucinated reasoning (states a rationale inconsistent with tool outputs) | Report the R_judge grounding rate as a named metric rather than omitting it |
| **LLM-judge/verifier bias or blind spots** (a real risk once clinician review is removed) | Use a different model family for generation vs. verification (§5.2); anchor primary correctness to DENTEX's fixed, dentist-verified ground truth rather than the judge's opinion alone; report judge-verifier agreement with ground truth on a subset as a sanity check |
| Cross-institution generalization untested | Evaluate zero-shot on the secondary 1,512-image dataset (different institution/equipment) as a generalization check (§6) |

---

## 9. Expected contributions (for the abstract/intro framing)

1. First tool-augmented, RL-fine-tuned agentic VLM applied to any medical or dental imaging task — extending the VTool-R1 recipe (validated only on general-domain chart/table QA) into a structured, clinically-grounded detection-diagnosis setting with a graded, multi-part reward.
2. First chain-of-thought + tool-use instruction dataset for panoramic dental radiograph diagnosis, generated and verified entirely via frontier-LLM APIs — a reusable resource, and a template pipeline other clinician-access-constrained teams can reuse.
3. Empirical ablation quantifying, separately, how much of the performance gain comes from tool use (H1) versus RL fine-tuning (H2) — a question the general medical-VLM literature (Med-R1) has flagged as unresolved even outside dentistry.
4. A fully automated, no-clinician-required validation protocol (dentist-verified benchmark ground truth + cross-model LLM-as-judge reasoning-grounding checks) that other resource-constrained medical-VLM projects can adopt as a credible substitute for a prospective expert study.

---

## 10. Status of open items

**Resolved in this version:**
- Backbone: Qwen3-VL-8B-Instruct (fallback 3B) — see §5.3.
- Training framework: TRL (single-GPU, LoRA/QLoRA) scaling to EasyR1 (multi-GPU) — see §5.4.
- Anomaly detection: explicitly cut from this paper's scope, deferred to a follow-up — see §5.6.
- **Clinical validation: deliberately does not require a clinician.** Primary validity anchor is DENTEX's own dentist-verified ground truth; reasoning quality is checked by a cross-model frontier-LLM judge; a prospective clinician study is named explicitly as future work, not hidden as a gap — see §6.
- **Compute plan: staged across available resources** — Kaggle/Colab free tier + laptop now, occasional RTX 4090 this semester, dedicated RTX 4090/server next semester — see §5.4 for exactly which stage each training step runs on.

**Still genuinely open (only you can resolve these):**
- Which two (or more) frontier LLM APIs to use as the generator/verifier pair in §5.2 (needs to be two different model families, and needs API budget planned for k-sample self-consistency generation across ~1,000 training cases).
- Exact current model versions at implementation time — re-check the model card/leaderboard right before you begin, since this space moves in months, not years.
