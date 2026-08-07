"""
Stage 1: Supervised Fine-Tuning (SFT) on Expert Multi-Turn Trajectories (§16, §17).

Includes:
- Prompt-loss masking builder (`build_sft_example`)
- Multi-trace expanding dataset (`TraceSFTDataset`, `DentalSFTDataset`)
- Trace dataset loader (`load_trace_dataset`)
- Full SFT training loop (`train_sft`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dental_agent.config import ProjectConfig, TrainingConfig
from dental_agent.model.backbone import load_model, apply_lora
from dental_agent.model.checkpoints import save_checkpoint
from dental_agent.agent.prompts import build_agent_system_prompt
from dental_agent.tools.registry import ToolRegistry


def build_sft_example(
    image: Image.Image,
    prompt_text_content: str,
    target_trace_text: str,
    processor: Any,
    system_prompt: str | None = None,
) -> dict[str, torch.Tensor]:
    """Build one training example: prompt tokens (system + user image/text) are
    masked out of the loss (label = -100); only the target trace's tokens contribute."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        def process_vision_info(msgs):
            return None, None

    if system_prompt is None:
        registry = ToolRegistry.create_default()
        system_prompt = build_agent_system_prompt(registry.format_tool_descriptions())

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text_content},
        ]},
    ]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + target_trace_text + (getattr(processor.tokenizer, "eos_token", None) or "<|im_end|>")

    image_inputs, video_inputs = process_vision_info(prompt_messages)
    prompt_enc = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    full_enc = processor(text=[full_text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    labels = full_enc["input_ids"].clone()
    prompt_len = prompt_enc["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    full_enc["labels"] = labels
    return dict(full_enc)


class TraceSFTDataset(Dataset):
    """Expands to one training example per verified trace (not per image) — an
    image with 3 verified traces contributes 3 training examples."""

    def __init__(
        self,
        trace_examples: list[dict[str, Any]],
        images_df: pd.DataFrame | None = None,
    ) -> None:
        self.samples: list[dict[str, Any]] = []
        for ex in trace_examples:
            img_path = ex.get("image_path")
            if not img_path and images_df is not None and "image_id" in ex:
                matches = images_df[images_df["id"] == ex["image_id"]]
                if not matches.empty:
                    img_path = matches.iloc[0]["local_path"]

            verified_traces = ex.get("verified_traces", [])
            if not verified_traces and "raw_trace" in ex:
                verified_traces = [ex["raw_trace"]]

            for trace_text in verified_traces:
                self.samples.append({
                    "image_path": img_path,
                    "image_id": ex.get("image_id"),
                    "prompt_text": f"Analyze this panoramic X-ray (image_id={ex.get('image_id', 'unknown')}).",
                    "target_text": trace_text if isinstance(trace_text, str) else json.dumps(trace_text),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class DentalSFTDataset(Dataset):
    """Dataset of verified expert multi-turn trajectories from JSON or JSONL file."""

    def __init__(self, data_path: str | Path, images_df: pd.DataFrame | None = None) -> None:
        self.samples: list[dict[str, Any]] = []
        path_str = str(data_path)
        if path_str.endswith(".jsonl"):
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        else:
            raw_list = load_trace_dataset(data_path)
            trace_ds = TraceSFTDataset(raw_list, images_df=images_df)
            self.samples = trace_ds.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def load_trace_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load whatever verified traces exist on disk (JSON or JSONL)."""
    default_dir = os.environ.get("DENTAL_AGENT_DATA_DIR", "data")
    path = path or os.path.join(default_dir, "pilot_traces.json")
    if not os.path.exists(str(path)):
        # Check jsonl alternative
        jsonl_alt = str(path).replace(".json", ".jsonl")
        if os.path.exists(jsonl_alt):
            path = jsonl_alt
        else:
            print(f"No trace file found at {path}")
            return []

    if str(path).endswith(".jsonl"):
        results = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    with open(path) as f:
        return json.load(f)


def train_sft(
    data_path: str | Path,
    images_df: pd.DataFrame | None = None,
    config: ProjectConfig | TrainingConfig | None = None,
    output_dir: str | Path = "checkpoints/sft",
    checkpoint_dir: str | Path = "checkpoints",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> str:
    """Execute Stage 1 SFT on verified expert traces with prompt-loss masking."""
    tr_cfg = config.training if isinstance(config, ProjectConfig) else (config or TrainingConfig())
    num_epochs = epochs or tr_cfg.sft_epochs
    lr = learning_rate or tr_cfg.sft_lr
    bsz = batch_size or 1
    grad_accum = 1

    print(f"--- Starting Stage 1 SFT Training (Epochs={num_epochs}, LR={lr}, BatchSize={bsz}) ---")

    model, processor = load_model(config)
    model = apply_lora(model, config)
    model.train()

    dataset = DentalSFTDataset(data_path, images_df=images_df)
    if len(dataset) == 0:
        raise ValueError(f"SFT dataset at {data_path} is empty.")

    registry = ToolRegistry.create_default()
    system_prompt = build_agent_system_prompt(registry.format_tool_descriptions())

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        pbar = tqdm(dataset, desc=f"SFT Epoch {epoch}/{num_epochs}")
        optimizer.zero_grad()

        for step, sample in enumerate(pbar):
            img_path = sample.get("image_path")
            if not img_path or not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")

            prompt_text = sample.get("prompt_text", "Analyze this panoramic X-ray.")
            target_text = sample.get("target_text") or sample.get("raw_trace", "")

            enc = build_sft_example(
                image=image,
                prompt_text_content=prompt_text,
                target_trace_text=target_text,
                processor=processor,
                system_prompt=system_prompt,
            )
            inputs = {k: v.to(model.device) for k, v in enc.items()}

            outputs = model(**inputs)
            loss = outputs.loss / grad_accum
            loss.backward()

            total_loss += loss.item() * grad_accum
            pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataset):
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(1, len(dataset))
        print(f"Epoch {epoch} finished. Average SFT Loss: {avg_loss:.4f}")

    saved_path = save_checkpoint(
        model=model,
        processor=processor,
        tag="sft-final",
        checkpoint_dir=checkpoint_dir,
        extra_metadata={"loss": avg_loss, "epochs": num_epochs},
    )
    print(f"Stage 1 SFT complete. Checkpoint saved to: {saved_path}")
    return saved_path


def sft_train(*args: Any, **kwargs: Any) -> str:
    """Stage 1 SFT training entrypoint (alias for train_sft)."""
    return train_sft(*args, **kwargs)


