# Visualization Prompts & Diagrams for VLM-DENTAL

This document contains publication-ready diagram definitions (Mermaid and Image Generation Prompts) for the **VLM-DENTAL** architecture, LangGraph reasoning loops, decoupled data engines, and dual-adapter reinforcement learning pipelines.

---

## 1. High-Level End-to-End System Architecture

### Diagram Specification (Mermaid)
```mermaid
flowchart TD
    subgraph S0["Stage 0: Grounding Model"]
        D0["DENTEX Dataset<br/>(BBoxes & FDI Labels)"] --> Y0["YOLOv8m Tooth Detector<br/>(5-Fold CV, mAP50=0.647)"]
        Y0 --> GT["locate_tooth Tool<br/>(Live in Agent Loop)"]
    end

    subgraph S1["Aim 1: Decoupled Synthetic Trace Engine"]
        OPG["Raw Panoramic Radiographs (OPG)"] --> LG["LangGraph Multi-Turn Loop<br/>(vLLM: Qwen/Qwen3.5-9B)"]
        GT -.->|Tooth Grounding| LG
        TR["Simulated Radiologist Tools<br/>(Zoom, Window, Denoise, Mirror)"] <-->|Dynamic Execution| LG
        LG --> RAW["Raw Unverified Traces<br/>(train_cot_traces_unverified.jsonl)"]
        RAW --> VP["ProviderPool Verifier<br/>(NVIDIA NIM / Groq / OpenRouter / Gemini)"]
        VP --> VER["Canonical Verified Traces<br/>(train_cot_traces.jsonl)"]
    end

    subgraph S2["Stage 1: Supervised Fine-Tuning (SFT)"]
        VER --> SFT_DATA["Conversational Multimodal Dataset<br/>(QwenVLDataCollator + Masked Loss)"]
        SFT_DATA --> SFT_TRAIN["QLoRA 4-Bit SFT Training<br/>(Base: Qwen/Qwen3.5-9B)"]
        SFT_TRAIN --> SFT_ADAPTER["SFT LoRA Adapter Weights<br/>(data/models/qwen3_5_9b_sft)"]
    end

    subgraph S3["Stage 2: GRPO Reinforcement Learning"]
        SFT_ADAPTER --> REF["Reference Model<br/>(Frozen SFT Adapter)"]
        SFT_ADAPTER --> POL["Active Policy<br/>(Trainable GRPO Adapter)"]
        POL --> ROLL["Multi-Trajectory Rollouts (G=4/8)"]
        ROLL --> REW["Composite Multi-Objective Reward<br/>(FDI + Diagnosis + Tool Validity + Format)"]
        REF --> KL["KL-Divergence Penalty"]
        REW & KL --> GRPO_OPT["GRPO Policy Update Step"]
        GRPO_OPT --> FINAL["Final Clinical Dental Agent"]
    end
```

### Image Generation Prompt (Publication Art)
> **Prompt:** Professional scientific diagram of an AI medical vision-language model training system for dental panoramic radiography (VLM-DENTAL). Dark high-tech clinical UI theme, glowing cyan and amber accents. Left side: Panoramic X-ray feeding into a multi-turn LangGraph agent with tool icons (magnifying glass zoom, contrast windowing, bilateral symmetry mirror, AI tooth detector). Middle: 2-stage decoupled generation pipeline with a local fast GPU server feeding into an asynchronous multi-cloud verifier pool. Right side: Dual-adapter LoRA reinforcement learning engine (GRPO) optimizing diagnostic accuracy and FDI tooth notation. Clean, isometric vector illustration, crisp typography, clean medical AI schematic, 8k resolution.

---

## 2. The 2-Stage Decoupled Trace Generation & Verification Engine

### Diagram Specification (Mermaid)
```mermaid
flowchart LR
    subgraph GEN["Stage A: High-Throughput Generation (Local vLLM)"]
        IMG["DENTEX Image + GT Target"] --> LG_NODE["LangGraph Reasoning Node<br/>(Qwen/Qwen3.5-9B via vLLM)"]
        LG_NODE <-->|Dynamic Tool Execution| EXEC["Deterministic Tools<br/>(zoom_crop, window_level, denoise, locate_tooth)"]
        LG_NODE --> JSONL_UNV["train_cot_traces_unverified.jsonl<br/>(Zero Rate-Limit, Full GPU Speed)"]
    end

    subgraph VER["Stage B: Asynchronous Cross-Family Verification (ProviderPool)"]
        JSONL_UNV --> QUEUE["Verification Queue<br/>(Resumable ID Tracking)"]
        QUEUE --> POOL["ProviderPool Load Balancer<br/>(300s Cooldown / 10 RPD Cap)"]
        POOL --> P1["NVIDIA NIM<br/>(meta/muse-glimmer-30b)"]
        POOL --> P2["Groq<br/>(qwen/qwen3.6-27b)"]
        POOL --> P3["OpenRouter<br/>(google/gemma-4-31b-it)"]
        POOL --> P4["Google Gemini<br/>(gemini-3.7-flash)"]
        P1 & P2 & P3 & P4 --> JUDGE{"Strict Grounding<br/>Verification"}
        JUDGE -->|PASS: Visually Grounded| CANONICAL["train_cot_traces.jsonl<br/>(Canonical SFT Dataset)"]
        JUDGE -->|FAIL: Hallucination| DROP["Rejected / Quarantined"]
    end
```

### Image Generation Prompt
> **Prompt:** Detailed technical flow diagram illustrating a decoupled two-stage data synthesis pipeline for AI training. Stage 1 on the left is a high-speed local GPU engine executing LangGraph multi-turn tool loops against dental X-rays at high frame rates. In the center, unverified JSONL traces queue up. Stage 2 on the right is a multi-cloud verification gateway round-robining across four external AI APIs (NVIDIA, Groq, OpenRouter, Google) with cooling timers and strict visual grounding checks, filtering into a verified golden dataset. Clean infographic style, futuristic blueprint aesthetic, sleek gradient colors on deep slate background.

---

## 3. LangGraph Interactive Multi-Turn Diagnostic Loop

### Diagram Specification (Mermaid)
```mermaid
stateDiagram-v2
    [*] --> InitialState: Load OPG Image & GT Target

    InitialState --> ReasoningNode: Build Context (Prompt + Prior Observations)
    
    state ReasoningNode {
        [*] --> CallVLM: Forward pass (Qwen/Qwen3.5-9B)
        CallVLM --> ParseJSON: Custom Bracket Parser
        ParseJSON --> DecisionTree
        DecisionTree --> ToolSelected: Valid Action & Args
        DecisionTree --> FinalDiagnosis: Valid final_answer Array
        DecisionTree --> SelfCorrection: Malformed JSON (Retry <= 3)
    }

    SelfCorrection --> ReasoningNode: Inject Repair Prompt
    ToolSelected --> ToolsNode: Route to ToolRegistry

    state ToolsNode {
        [*] --> DispatchTool
        DispatchTool --> ZoomCrop: zoom_crop(bbox)
        DispatchTool --> WindowLevel: window_level(preset='bone'/'enamel')
        DispatchTool --> BilateralDenoise: denoise(method='bilateral')
        DispatchTool --> Contralateral: contralateral_compare(bbox, quad)
        DispatchTool --> GroundingDetector: locate_tooth(fdi_number)
        
        ZoomCrop --> PackObservation
        WindowLevel --> PackObservation
        BilateralDenoise --> PackObservation
        Contralateral --> PackObservation
        GroundingDetector --> PackObservation
        PackObservation --> UpdateMessages: Append Visual / Text Result
    }

    ToolsNode --> ReasoningNode: Next Turn (Observation Context)
    FinalDiagnosis --> VerificationStep: Output Full Diagnostic Trace
    VerificationStep --> [*]: Passed to ProviderPool
```

### Image Generation Prompt
> **Prompt:** Futuristic UI workflow visualization of an autonomous medical AI agent's LangGraph decision cycle. Circular agent reasoning loop showing state transitions: Observation of panoramic dental X-ray -> Multi-modal reasoning turn -> Tool selection branching into specialized medical tools (magnification loupe, bone contrast window, symmetry mirror, AI tooth detector) -> Real dynamic execution returning enhanced visual crops -> Convergence into structured FDI dental diagnostic report. Modern dark theme with cyan, violet, and emerald glowing nodes, ultra-clean UI design.

---

## 4. Simulated Radiologist Tool Suite Architecture

### Diagram Specification (Mermaid)
```mermaid
flowchart TD
    BASE["Base Panoramic Radiograph<br/>(Full Uncropped Image)"] --> REG["ToolRegistry<br/>(Dynamic Python Execution)"]
    
    REG --> T1["zoom_crop(bbox, padding=0.15)<br/>Extracts high-res pathology region"]
    REG --> T2["window_level(preset='bone'|'enamel')<br/>Non-linear intensity windowing"]
    REG --> T3["denoise(method='bilateral')<br/>Edge-preserving noise reduction"]
    REG --> T4["contralateral_compare(bbox, quadrant)<br/>Anatomical bilateral symmetry mirror crop"]
    REG --> T5["locate_tooth(fdi_number)<br/>YOLOv8m 5-Fold Grounding Model"]
    
    T1 --> OBS["Multimodal Observation Payload<br/>(PIL Image + FDI Coordinate Metadata)"]
    T2 --> OBS
    T3 --> OBS
    T4 --> OBS
    T5 --> OBS
    
    OBS --> VLM["VLM Agent Context<br/>(Next Reasoning Turn)"]
```

---

## 5. Dual-Adapter GRPO Reinforcement Learning Architecture

### Diagram Specification (Mermaid)
```mermaid
flowchart TD
    subgraph MEM["Single 4-Bit Base Model in GPU VRAM (Qwen/Qwen3.5-9B)"]
        BASE_WEIGHTS["Quantized 4-Bit Base Weights<br/>(Frozen, ~6GB VRAM)"]
        ADAPT_REF["Adapter A: Frozen SFT Reference<br/>(LoRA: reference)"]
        ADAPT_POL["Adapter B: Trainable GRPO Policy<br/>(LoRA: grpo_policy)"]
    end

    ENV["LangGraph Dynamic Environment<br/>(X-Ray + ToolRegistry)"] --> SAMP["Sample G=4 or G=8 Rollouts"]
    ADAPT_POL --> SAMP

    SAMP --> LOGP_POL["Log-probs Policy: π_θ(a_t | s_t)"]
    SAMP --> TOGGLE["PEFT Toggle Adapter<br/>set_adapter('reference')"]
    TOGGLE --> LOGP_REF["Log-probs Ref: π_ref(a_t | s_t)"]

    SAMP --> R_EVAL["Multi-Objective Reward Function"]
    R_EVAL --> R1["FDI Accuracy Reward (R_FDI)"]
    R_EVAL --> R2["Pathology Diagnosis Reward (R_Diag)"]
    R_EVAL --> R3["Tool Validity & Utility (R_Tool)"]
    R_EVAL --> R4["Format & JSON Integrity (R_Format)"]
    
    R1 & R2 & R3 & R4 --> R_TOTAL["Unified Reward Score R_i"]
    R_TOTAL --> ADV["Group Advantage A_i = (R_i - μ) / σ"]
    
    LOGP_POL & LOGP_REF & ADV --> LOSS["GRPO Loss Objective<br/>min E[ -min(r_t A_t, clip(r_t) A_t) + β KL(π_θ || π_ref) ]"]
    LOSS --> OPT["AdamW Optimizer Update on Adapter B"]
```

