"""
Stage 1: Supervised Fine-Tuning (SFT) on Expert Multi-Turn Trajectories (§16, §17).

Production implementation supporting:
- Strict Track Segregation: Track A (with_tools) vs Track B (no_tools)
- Dynamic Image Path Resolution across Cloud / Kaggle / Local platforms
- Conversational Assistant-Only Loss Masking (<|im_start|>assistant ... <|im_end|>)
- Bucketed Collator with Static Discrete Snapping & Strict Right-Padding
- Native PyTorch/XLA (Cloud TPU v5e-8) and Multi-GPU Accelerate Execution
"""

from __future__ import annotations

import json
import os
import glob
from pathlib import Path
from typing import Any, List, Dict, Optional
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dental_agent.config import ProjectConfig, TrainingConfig
from dental_agent.model.backbone import load_model, apply_lora, safe_process_vision_info
from dental_agent.model.checkpoints import save_checkpoint
from dental_agent.tools.registry import ToolRegistry
from dental_agent.agent.tool_dispatch import execute_tool_call


_IMAGE_PATH_RESOLVE_CACHE: dict[tuple[str, str, str], str] = {}


def resolve_image_path(sample: dict[str, Any], data_dir: str | Path = "data") -> Optional[str]:
    """Dynamically resolve image path across Kaggle, Hugging Face snapshots, and local directories.

    Aligns directly with canonical Hugging Face dataset layouts and Kaggle mounts:
    - DENTEX (Reza-Nadimi/dentex-train-images): images/{id}.png
    - Tufts (Reza-Nadimi/tufts-train-images): Radiographs/{id}.JPG
    """
    raw_path = sample.get("image_path")
    if raw_path and os.path.isfile(raw_path):
        return raw_path

    image_id = sample.get("image_id")
    if image_id is None:
        return None

    str_id = str(image_id)
    ds_name = (sample.get("dataset") or "").lower().strip()
    origin_file = str(sample.get("_origin_file", "")).lower()
    if not ds_name:
        if "tufts" in origin_file:
            ds_name = "tufts"
        elif "dentex" in origin_file:
            ds_name = "dentex"

    cache_key = (ds_name, str_id, str(data_dir))
    if cache_key in _IMAGE_PATH_RESOLVE_CACHE:
        return _IMAGE_PATH_RESOLVE_CACHE[cache_key]

    base = Path(data_dir)
    candidates: list[Path] = []
    k_base = Path("/kaggle/input") if os.path.exists("/kaggle/input") else None

    if ds_name == "dentex":
        candidates.extend([
            base / "dentex" / "images" / f"{str_id}.png",
            base / "dentex" / f"{str_id}.png",
            base / "images" / f"{str_id}.png",
        ])
        if k_base:
            candidates.extend([
                k_base / "dentex" / "images" / f"{str_id}.png",
                k_base / "dentex-panoramic" / "images" / f"{str_id}.png",
                k_base / "datasets" / "rezanadimikj" / "dentex-panoramic" / "images" / f"{str_id}.png",
            ])
    elif ds_name == "tufts":
        candidates.extend([
            base / "tufts" / "Radiographs" / f"{str_id}.JPG",
            base / "tufts" / "Radiographs" / f"{str_id}.jpg",
            base / "Tufts" / "Radiographs" / f"{str_id}.JPG",
            base / "Tufts" / "Radiographs" / f"{str_id}.jpg",
            base / "tufts" / "images" / f"{str_id}.png",
        ])
        if k_base:
            candidates.extend([
                k_base / "tufts" / "Radiographs" / f"{str_id}.JPG",
                k_base / "tufts" / "Radiographs" / f"{str_id}.jpg",
                k_base / "tufts-panoramic" / "Radiographs" / f"{str_id}.JPG",
                k_base / "tufts-panoramic" / "Radiographs" / f"{str_id}.jpg",
                k_base / "datasets" / "rezanadimikj" / "tufts-panoramic" / "Radiographs" / f"{str_id}.JPG",
            ])
    else:
        candidates.extend([
            base / "images" / f"{str_id}.png",
            base / "dentex" / "images" / f"{str_id}.png",
            base / "tufts" / "Radiographs" / f"{str_id}.JPG",
            base / "tufts" / "Radiographs" / f"{str_id}.jpg",
        ])

    for cand in candidates:
        if cand.is_file():
            res = str(cand)
            _IMAGE_PATH_RESOLVE_CACHE[cache_key] = res
            return res

    # Glob fallback within respective dataset folder
    search_root = base / ("tufts" if ds_name == "tufts" else "dentex")
    if not search_root.exists():
        search_root = base
    patterns = [f"**/{str_id}.*", f"**/train_{str_id}.*"] if ds_name == "dentex" else [f"**/{str_id}.*"]
    for pat in patterns:
        for match in search_root.glob(pat):
            if match.is_file() and match.suffix.lower() in (".png", ".jpg", ".jpeg"):
                res = str(match)
                _IMAGE_PATH_RESOLVE_CACHE[cache_key] = res
                return res

    return None


def build_conversational_labels(
    input_ids: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    """Build assistant-only loss mask for conversational Qwen-VL sequences.

    Tokens between `<|im_start|>assistant\n` and `<|im_end|>` retain their token IDs.
    All system prompts, user queries, tool return observations, and padding tokens
    are masked with `labels = -100`.
    """
    labels = torch.full_like(input_ids, -100)
    flat_ids = input_ids[0].tolist() if input_ids.dim() == 2 else input_ids.tolist()

    # Identify special token sequences
    im_start_id = getattr(tokenizer, "im_start_id", None)
    if not isinstance(im_start_id, int):
        enc_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_start_id = enc_start[0] if enc_start else None

    im_end_id = getattr(tokenizer, "im_end_id", None)
    if not isinstance(im_end_id, int):
        enc_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        im_end_id = enc_end[0] if enc_end else None

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

    # Context-aware extraction: capture exact token sequence of assistant inside <|im_start|>assistant\n
    # Guard against BPE tokenizers encoding words differently in isolation vs in-context (Claude Point 4)
    contextual_asst_ids = None
    try:
        header_enc = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        if header_enc and im_start_id is not None and header_enc[0] == im_start_id:
            nl_pos = -1
            for idx in range(len(header_enc) - 1, 0, -1):
                if header_enc[idx] == newline_id:
                    nl_pos = idx
                    break
            if nl_pos > 1:
                contextual_asst_ids = header_enc[1:nl_pos]
    except Exception:
        contextual_asst_ids = None

    candidate_patterns = [p for p in [assistant_token_ids, contextual_asst_ids] if p]

    i = 0
    seq_len = len(flat_ids)
    while i < seq_len:
        # Match <|im_start|> followed by assistant token pattern
        if flat_ids[i] == im_start_id:
            matched_len = None
            for pat in candidate_patterns:
                if flat_ids[i + 1 : i + 1 + len(pat)] == pat:
                    matched_len = len(pat)
                    break

            if matched_len is not None:
                # Assistant turn found! Skip past 'assistant\n'
                start_idx = i + 1 + matched_len
                if start_idx < seq_len and flat_ids[start_idx] == newline_id:
                    start_idx += 1

                # Search forward for matching <|im_end|>
                end_idx = start_idx
                while end_idx < seq_len and flat_ids[end_idx] != im_end_id:
                    end_idx += 1

                # Include the <|im_end|> token so model learns when to stop
                if end_idx < seq_len:
                    end_idx += 1

                if input_ids.dim() == 2:
                    labels[0, start_idx:end_idx] = input_ids[0, start_idx:end_idx]
                else:
                    labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

                i = end_idx
                continue
        i += 1

    return labels


class BucketedQwenVLCollator:
    """Collates variable-length multimodal examples into static discrete buckets.

    Enforces:
    1. Strict right-padding (`padding_side = "right"`) to preserve 3D MRoPE coordinate origin.
    2. Discrete bucket snapping up to 16,384 tokens to eliminate XLA dynamic graph recompilations.
    3. Overlength warning diagnostics when sequences exceed maximum headroom.
    4. Padding tokens masked with `labels = -100`.
    """

    BUCKETS_WITH_TOOLS = [8192, 16384, 32768]
    BUCKETS_NO_TOOLS = [1536, 2048, 2560, 3072, 8192]

    def __init__(
        self,
        processor: Any,
        track: str = "with_tools",
        max_seq_len: int | None = None,
        custom_buckets: list[int] | None = None,
        dynamic_padding: bool = False,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.track = track
        self.dynamic_padding = dynamic_padding
        self.max_seq_len = max_seq_len
        if custom_buckets:
            self.buckets = sorted(custom_buckets)
        elif track == "no_tools":
            self.buckets = self.BUCKETS_NO_TOOLS
        else:
            self.buckets = self.BUCKETS_WITH_TOOLS

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        self.pad_token_id = pad_id if pad_id is not None else 0

    def _snap_to_bucket(self, length: int) -> int:
        for b in self.buckets:
            if length <= b:
                return b
        return self.buckets[-1]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}

        # Determine target sequence length: dynamic (GPU/CPU eager mode) vs static uniform length (Cloud TPU v5e-8 XLA)
        max_batch_len = max(ex["input_ids"].shape[1] for ex in batch)
        if self.dynamic_padding:
            target_len = max_batch_len
        elif self.max_seq_len:
            target_len = self.max_seq_len
        else:
            target_len = self._snap_to_bucket(max_batch_len)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        padded_mm_token_type_ids = []

        all_pixel_values = []
        all_image_grid_thw = []
        all_pixel_values_videos = []
        all_video_grid_thw = []

        has_mm_types = any("mm_token_type_ids" in ex and ex["mm_token_type_ids"] is not None for ex in batch)
        has_multimodal = any(
            ("image_grid_thw" in ex and ex["image_grid_thw"] is not None)
            or ("video_grid_thw" in ex and ex["video_grid_thw"] is not None)
            for ex in batch
        )

        for ex in batch:
            curr_len = ex["input_ids"].shape[1]
            if curr_len > target_len:
                import warnings
                warnings.warn(
                    f"[COLLATOR WARNING] Sequence token length ({curr_len}) exceeds maximum bucket ({target_len}). "
                    f"Truncating tail tokens; note that this may truncate the assistant's final diagnostic response. "
                    "Consider checking upstream tool-call image resolution or splitting turns.",
                    UserWarning,
                    stacklevel=2,
                )
                # Truncate if exceeding maximum bucket
                input_ids = ex["input_ids"][:, :target_len]
                labels = ex["labels"][:, :target_len]
                attention_mask = ex.get("attention_mask", torch.ones_like(ex["input_ids"]))[:, :target_len]
                pad_needed = 0
            else:
                input_ids = ex["input_ids"]
                labels = ex["labels"]
                attention_mask = ex.get("attention_mask", torch.ones_like(ex["input_ids"]))
                pad_needed = target_len - curr_len

            if pad_needed > 0:
                pad_tokens = torch.full((1, pad_needed), self.pad_token_id, dtype=input_ids.dtype)
                pad_labels = torch.full((1, pad_needed), -100, dtype=labels.dtype)
                pad_mask = torch.zeros((1, pad_needed), dtype=attention_mask.dtype)

                # Right padding invariant for 3D MRoPE
                input_ids = torch.cat([input_ids, pad_tokens], dim=1)
                labels = torch.cat([labels, pad_labels], dim=1)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

            # Preserve and right-pad mm_token_type_ids (required for M-RoPE 3D position computation in Qwen3.5)
            mm_types = ex.get("mm_token_type_ids")
            if mm_types is not None:
                if mm_types.dim() == 1:
                    mm_types = mm_types.unsqueeze(0)
                if curr_len > target_len:
                    mm_types = mm_types[:, :target_len]
                if pad_needed > 0:
                    pad_mm = torch.zeros((1, pad_needed), dtype=mm_types.dtype, device=mm_types.device)
                    mm_types = torch.cat([mm_types, pad_mm], dim=1)
                padded_mm_token_type_ids.append(mm_types)
            elif has_multimodal or has_mm_types:
                # Defensive synthesis of mm_token_type_ids: 0 for text/pad, 1 for image, 2 for video
                mm_types = torch.zeros_like(input_ids)
                image_token_id = getattr(self.processor, "image_token_id", None)
                if not isinstance(image_token_id, int) and hasattr(self.processor, "tokenizer"):
                    image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
                if not isinstance(image_token_id, int) and hasattr(self.processor, "tokenizer"):
                    try:
                        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    except Exception:
                        image_token_id = None
                if isinstance(image_token_id, int) and image_token_id > 0:
                    mm_types[input_ids == image_token_id] = 1
                padded_mm_token_type_ids.append(mm_types)

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)

            if "pixel_values" in ex and ex["pixel_values"] is not None:
                all_pixel_values.append(ex["pixel_values"])
            if "image_grid_thw" in ex and ex["image_grid_thw"] is not None:
                all_image_grid_thw.append(ex["image_grid_thw"])
            if "pixel_values_videos" in ex and ex["pixel_values_videos"] is not None:
                all_pixel_values_videos.append(ex["pixel_values_videos"])
            if "video_grid_thw" in ex and ex["video_grid_thw"] is not None:
                all_video_grid_thw.append(ex["video_grid_thw"])

        collated = {
            "input_ids": torch.cat(padded_input_ids, dim=0),
            "attention_mask": torch.cat(padded_attention_mask, dim=0),
            "labels": torch.cat(padded_labels, dim=0),
        }

        if padded_mm_token_type_ids:
            collated["mm_token_type_ids"] = torch.cat(padded_mm_token_type_ids, dim=0)
        if all_pixel_values:
            collated["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            collated["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        if all_pixel_values_videos:
            collated["pixel_values_videos"] = torch.cat(all_pixel_values_videos, dim=0)
        if all_video_grid_thw:
            collated["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)

        return collated


class DentalSFTDataset(Dataset):
    """Production SFT dataset ingesting multi-turn or direct CoT traces."""

    def __init__(
        self,
        data_path: str | Path | list[str | Path],
        processor: Any,
        track: str = "with_tools",
        data_dir: str | Path = "data",
        max_seq_len: int | None = None,
        token_lengths_manifest: str | Path | None = None,
    ) -> None:
        self.processor = processor
        self.track = track
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.records: list[dict[str, Any]] = []
        self.registry = ToolRegistry.create_default()
        self._crop_cache: dict[str, Image.Image] = {}

        # Resolve pre-computed token lengths manifest for instant O(1) filtering on Kaggle/Colab
        token_map: dict[str, int] = {}
        manifest_candidates = []
        if token_lengths_manifest:
            manifest_candidates.append(Path(token_lengths_manifest))
        manifest_candidates.extend([
            Path(data_dir) / "traces" / "trace_token_lengths.json",
            Path("data/traces/trace_token_lengths.json"),
            Path("/kaggle/working/data/traces/trace_token_lengths.json"),
        ])
        for mc in manifest_candidates:
            if mc.is_file():
                try:
                    with open(mc, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                        token_map = m_data.get("lengths_by_file_and_id", {})
                        if not token_map:
                            token_map = m_data.get("lengths_by_image_id", {})
                    if token_map:
                        break
                except Exception:
                    pass

        raw_records: list[dict[str, Any]] = []
        paths = [data_path] if isinstance(data_path, (str, Path)) else list(data_path)
        for p in paths:
            p_obj = Path(p)
            if not p_obj.is_file():
                raise FileNotFoundError(f"SFT trace file not found: {p_obj}")
            fname = p_obj.name
            with open(p_obj, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        r["_origin_file"] = fname
                        raw_records.append(r)

        if max_seq_len is not None and token_map:
            retained: list[dict[str, Any]] = []
            filtered_count = 0
            for r in raw_records:
                rid = str(r.get("image_id", ""))
                qkey = f"{r.get('_origin_file', '')}::{rid}"
                tok_len = token_map.get(qkey)
                if tok_len is None:
                    tok_len = token_map.get(rid)
                if tok_len is not None and tok_len > max_seq_len:
                    filtered_count += 1
                else:
                    retained.append(r)
            self.records = retained
            pct = (len(retained) / max(len(raw_records), 1)) * 100
            print(
                f"[DATASET MASK] Evaluated {len(raw_records)} traces: retained {len(retained)} "
                f"(<= {max_seq_len} tokens, {pct:.1f}%), filtered out {filtered_count} overlength traces."
            )
        else:
            self.records = raw_records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image_path = resolve_image_path(rec, self.data_dir)
        if not image_path:
            # Create a 512x512 dummy neutral image if file is completely missing
            base_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        else:
            base_image = Image.open(image_path).convert("RGB")

        raw_messages = rec.get("messages", [])
        if not raw_messages:
            # Synthesize direct CoT structure if messages list is omitted
            raw_messages = [
                {"role": "system", "content": "You are an expert dental radiologist AI."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base_image},
                        {"type": "text", "text": f"Analyze panoramic X-ray (image_id={rec.get('image_id')})."},
                    ],
                },
                {"role": "assistant", "content": json.dumps(rec.get("final_answer", {}))},
            ]

        # Process raw messages with dynamic tool image generation
        sanitized_messages = []
        turns_records = rec.get("turns", [])
        image_id_str = str(rec.get("image_id", idx))

        IMAGE_PRODUCING_TOOLS = {
            "zoom_crop",
            "denoise",
            "window_level",
            "contralateral_compare",
            "enhance_contrast",
        }

        last_assistant_tool_calls: list[dict[str, Any]] = []
        assistant_turn_count = 0

        for msg_idx, msg in enumerate(raw_messages):
            role = msg.get("role")
            content = msg.get("content")

            if role == "assistant":
                last_assistant_tool_calls = []
                try:
                    if isinstance(content, str):
                        parsed = json.loads(content)
                    elif isinstance(content, dict):
                        parsed = content
                    else:
                        parsed = {}
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        calls = parsed["tool_calls"]
                        if isinstance(calls, list):
                            last_assistant_tool_calls = calls
                except Exception:
                    pass
                assistant_turn_count += 1
                sanitized_messages.append(msg)

            elif role == "user" and (msg_idx == 1 or len(sanitized_messages) <= 1):
                # Turn 1: Always provide the authentic full native-resolution panoramic base_image
                sanitized_content = []
                if isinstance(content, list):
                    prompt_text = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = item.get("text", "")
                            if "[Earlier tool result omitted" not in txt:
                                prompt_text += txt + "\n"
                    if not prompt_text.strip():
                        prompt_text = "Analyze this panoramic X-ray. Identify any abnormal teeth and determine the diagnosis."
                    sanitized_content.append({"type": "image", "image": base_image})
                    sanitized_content.append({"type": "text", "text": prompt_text.strip()})
                elif isinstance(content, str):
                    sanitized_content.append({"type": "image", "image": base_image})
                    sanitized_content.append({"type": "text", "text": content})
                else:
                    sanitized_content.append({"type": "image", "image": base_image})
                sanitized_messages.append({"role": "user", "content": sanitized_content})

            elif role == "user":
                # Subsequent tool observation turns: generate authentic tool outputs dynamically
                if isinstance(content, list):
                    sanitized_content = []
                    pending_calls = list(last_assistant_tool_calls)
                    if not pending_calls and assistant_turn_count - 1 < len(turns_records):
                        t_rec = turns_records[assistant_turn_count - 1]
                        raw_calls = t_rec.get("tool_calls_this_turn", [])
                        pending_calls = [{"tool": c.get("tool_name"), "args": c.get("tool_args", {})} for c in raw_calls]

                    call_cursor = 0
                    for item_idx, item in enumerate(content):
                        if isinstance(item, dict) and item.get("type") == "image":
                            # Match the tool that generated this image
                            matched_tool_name = None
                            if item_idx + 1 < len(content):
                                next_item = content[item_idx + 1]
                                if isinstance(next_item, dict) and next_item.get("type") == "text":
                                    txt = next_item.get("text", "")
                                    if txt.startswith("Result of "):
                                        matched_tool_name = txt.split(":")[0].replace("Result of ", "").strip()

                            tool_args = {}
                            if matched_tool_name:
                                for pc in pending_calls:
                                    if pc.get("tool") == matched_tool_name:
                                        tool_args = pc.get("args", {})
                                        break
                            elif call_cursor < len(pending_calls):
                                pc = pending_calls[call_cursor]
                                matched_tool_name = pc.get("tool")
                                tool_args = pc.get("args", {})
                                call_cursor += 1

                            if matched_tool_name is None:
                                matched_tool_name = "zoom_crop"

                            tool_img = None
                            if matched_tool_name in IMAGE_PRODUCING_TOOLS:
                                cache_key = f"{image_id_str}_{matched_tool_name}_{json.dumps(tool_args, sort_keys=True)}"
                                if cache_key in self._crop_cache:
                                    tool_img = self._crop_cache[cache_key]
                                else:
                                    try:
                                        res = execute_tool_call(self.registry, matched_tool_name, tool_args, base_image)
                                        if isinstance(res, Image.Image):
                                            tool_img = res
                                            self._crop_cache[cache_key] = tool_img
                                    except Exception:
                                        pass

                            if tool_img is None:
                                try:
                                    fallback_box = tool_args.get("bbox") if isinstance(tool_args, dict) and "bbox" in tool_args else [100.0, 100.0, 200.0, 200.0]
                                    tool_img = execute_tool_call(self.registry, "zoom_crop", {"bbox": fallback_box}, base_image)
                                except Exception:
                                    tool_img = Image.new("RGB", (256, 256), color=(128, 128, 128))

                            sanitized_content.append({"type": "image", "image": tool_img})
                        else:
                            sanitized_content.append(item)

                    sanitized_messages.append({"role": "user", "content": sanitized_content})
                else:
                    sanitized_messages.append(msg)

            else:
                sanitized_messages.append(msg)

        text = self.processor.apply_chat_template(
            sanitized_messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = safe_process_vision_info(sanitized_messages)

        enc = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # Apply conversational assistant-only loss masking
        labels = build_conversational_labels(enc["input_ids"], self.processor.tokenizer)
        enc["labels"] = labels

        # Defensive assertion (Claude Point 4): guard against zero supervision
        num_supervised = (labels != -100).sum().item()
        if num_supervised == 0:
            raise ValueError(
                f"[LOSS MASKING ERROR] Zero supervised tokens found for sample (image_id={rec.get('image_id')})! "
                "The assistant turn delimiter was not matched by build_conversational_labels. "
                "Failing fast to prevent training on empty supervision."
            )

        return {k: v for k, v in enc.items()}


def unwrap_peft_model(model: Any) -> Any:
    """Safely unwrap a model wrapped in FSDP, DDP, or DataParallel down to the underlying PeftModel/PreTrainedModel.

    Guarantees that attribute accesses like .set_adapter(), .save_pretrained(),
    and .peft_config target the underlying model rather than a distributed wrapper.
    """
    cur = model
    while hasattr(cur, "module"):
        cur = cur.module
    return cur


def wrap_distributed_model(
    model: Any,
    is_tpu: bool = False,
    num_cores: int = 1,
    use_fsdp: bool = True,
    is_master: bool = True,
    use_spmd: bool = False,
) -> Any:
    """Wrap model for distributed TPU v5e-8 or multi-GPU execution.

    On Google Cloud TPU v5e-8, each core has only 16 GB HBM. A 9B model in BF16
    is ~18.4 GB just for frozen base weights. When `is_tpu and num_cores > 1 and use_fsdp`,
    this function wraps the model with `torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`
    (FSDP) with `reshard_after_forward=True`. This shards the 18.4 GB model across the 8 cores
    (~2.3 GB per core), keeping per-core memory footprint comfortably at ~4-5 GB and protecting
    against the 16 GB HBM ceiling.
    """
    if is_tpu:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except Exception as e:
            if is_master:
                print(f"[FSDP WARNING] torch_xla is not installed or available ({e}); skipping TPU placement.")
            return model

        if num_cores > 1 and use_spmd:
            try:
                import os
                import numpy as np
                import torch_xla.distributed.spmd as xs
                import torch_xla.runtime as xr

                os.environ["XLA_USE_SPMD"] = "1"
                num_devices = xr.global_device_count()
                device_ids = np.arange(num_devices)
                mesh = xs.Mesh(device_ids, (num_devices,), ("data",))
                xs.set_global_mesh(mesh)
                if is_master:
                    print(f"[SPMD] Initialized GSPMD 1D mesh across {num_devices} TPU devices.")

                model = model.to(device)

                # Shard parameter tensors across the TPU devices so each core only holds ~2.3 GB HBM
                sharded_count = 0
                for name, p in model.named_parameters():
                    if p.ndim >= 2 and p.shape[0] % num_devices == 0:
                        xs.mark_sharding(p, mesh, ("data",) + (None,) * (p.ndim - 1))
                        sharded_count += 1
                    elif p.ndim >= 2 and p.shape[1] % num_devices == 0:
                        xs.mark_sharding(p, mesh, (None, "data") + (None,) * (p.ndim - 2))
                        sharded_count += 1
                    elif p.ndim == 1 and p.shape[0] % num_devices == 0:
                        xs.mark_sharding(p, mesh, ("data",))
                        sharded_count += 1

                if is_master:
                    print(f"[SPMD] Sharded {sharded_count} parameter tensors across {num_devices} TPU cores via xs.mark_sharding.")
                return model
            except Exception as e:
                if is_master:
                    print(f"[SPMD ERROR] GSPMD setup failed: {e}")
                raise RuntimeError(f"PyTorch/XLA SPMD initialization failed: {e}") from e

        if num_cores > 1 and use_fsdp and not use_spmd:
            try:
                from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

                # PyTorch/XLA FSDP strictly requires master parameters in torch.float32 for sharding.
                # If not already float32, cast the model parameters before wrapping.
                if next(model.parameters()).dtype != torch.float32:
                    model = model.float()

                auto_wrap_policy = None
                try:
                    from peft.utils.other import fsdp_auto_wrap_policy
                    raw_policy = fsdp_auto_wrap_policy(model)
                    if raw_policy is not None:
                        # PyTorch/XLA's FSDP recursive_wrap calls auto_wrap_policy(module, recurse=True, unwrapped_params=num_params).
                        # PEFT's policy uses PyTorch core's _or_policy, which expects parameter name 'nonwrapped_numel'.
                        # This adapter translates unwrapped_params -> nonwrapped_numel to prevent TypeError: unexpected keyword argument 'unwrapped_params'.
                        def xla_policy(module, recurse, unwrapped_params=0, **kwargs):
                            try:
                                return raw_policy(module=module, recurse=recurse, nonwrapped_numel=unwrapped_params)
                            except TypeError:
                                try:
                                    return raw_policy(module, recurse, unwrapped_params)
                                except TypeError:
                                    return raw_policy(module=module, recurse=recurse)

                        auto_wrap_policy = xla_policy
                except Exception as e:
                    if is_master:
                        print(f"[FSDP] PEFT auto_wrap_policy detection failed ({e}); falling back to native transformer policy.")

                # Fallback to torch_xla native transformer_auto_wrap_policy if PEFT policy is not available
                if auto_wrap_policy is None:
                    try:
                        from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
                        from functools import partial
                        transformer_cls = set()
                        for m in model.modules():
                            cls_name = m.__class__.__name__
                            if any(k in cls_name for k in ["DecoderLayer", "Block", "TransformerLayer"]):
                                transformer_cls.add(m.__class__)
                        if transformer_cls:
                            base_t_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls)
                            def xla_transformer_policy(module, recurse, unwrapped_params=0, **kwargs):
                                try:
                                    return base_t_policy(module=module, recurse=recurse, nonwrapped_numel=unwrapped_params)
                                except TypeError:
                                    return base_t_policy(module=module, recurse=recurse, unwrapped_params=unwrapped_params)
                            auto_wrap_policy = xla_transformer_policy
                    except Exception:
                        pass

                wrap_kwargs: dict[str, Any] = {
                    "reshard_after_forward": True,
                    "compute_dtype": torch.bfloat16,
                    "buffer_dtype": torch.bfloat16,
                }
                if auto_wrap_policy is not None:
                    wrap_kwargs["auto_wrap_policy"] = auto_wrap_policy

                model = FSDP(model, **wrap_kwargs)
                if is_master:
                    per_core_gb = 36.0 / max(num_cores, 1)
                    print(
                        f"[FSDP] Successfully wrapped model in torch_xla XlaFullyShardedDataParallel across {num_cores} TPU cores."
                    )
                    print(
                        f"[FSDP] Base model parameters sharded in FP32 with BF16 compute (~{per_core_gb:.2f} GB/core). 16 GB HBM ceiling protected."
                    )
                return model
            except Exception as e:
                raise RuntimeError(
                    f"FSDP parameter sharding failed across {num_cores} TPU cores: {e}. "
                    f"Cannot fall back to single-device model.to(device) because a 9B BF16 model (~18.4 GB) "
                    f"exceeds the 16 GB per-core HBM limit of Cloud TPU v5e-8."
                ) from e
        else:
            return model.to(device)

    # CUDA / CPU device placement
    if torch.cuda.is_available() and not hasattr(model, "hf_device_map"):
        return model.to("cuda")
    return model


def train_sft(
    data_path: str | Path,
    track: str = "with_tools",
    config: ProjectConfig | TrainingConfig | None = None,
    output_dir: str | Path = "data/models/qwen3_5_9b_sft_tools",
    checkpoint_dir: str | Path = "data/models",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-5,
    precision: str = "bf16",
    gradient_accumulation_steps: int = 16,
    hf_repo: str | None = None,
    push_every_steps: int = 25,
    num_cores: int = 1,
    use_fsdp: bool = True,
    use_spmd: bool = False,
    max_seq_len: int = 32768,
) -> str:
    """Execute Stage 1 SFT on verified expert traces with conversational loss masking."""
    print(f"--- Starting Stage 1 SFT Training (Track={track}, Epochs={epochs}, LR={learning_rate}) ---")

    # Load model and tokenizer
    model, processor = load_model(config)
    model = apply_lora(model, config)

    is_tpu = False
    try:
        import torch_xla.core.xla_model as xm
        is_tpu = True
    except Exception:
        pass

    model = wrap_distributed_model(
        model,
        is_tpu=is_tpu,
        num_cores=num_cores,
        use_fsdp=use_fsdp,
        use_spmd=use_spmd,
    )
    model.train()

    dataset = DentalSFTDataset(data_path, processor=processor, track=track, max_seq_len=max_seq_len)
    if len(dataset) == 0:
        raise ValueError(f"SFT dataset at {data_path} is empty.")

    collator = BucketedQwenVLCollator(processor, track=track, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = 0
    log_file = Path(output_dir) / "training_loss.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch}/{epochs}")
        optimizer.zero_grad()
        accum_loss_sum = 0.0
        accum_valid_tokens = 0

        for step, batch in enumerate(pbar):
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**inputs)

            # Mathematical Gradient Accumulation (Claude Point 1):
            # outputs.loss is HF internal mean across valid tokens in this microbatch.
            # Multiply by num_valid to get unreduced loss sum, accumulate gradients,
            # and divide by total valid tokens at the accumulation boundary.
            valid_tokens = (inputs["labels"] != -100).sum()
            num_valid = valid_tokens.item()

            if num_valid > 0:
                batch_loss_sum = outputs.loss * valid_tokens
                batch_loss_sum.backward()
                accum_loss_sum += batch_loss_sum.item()
                accum_valid_tokens += num_valid

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                if accum_valid_tokens > 0:
                    scale = 1.0 / max(float(accum_valid_tokens), 1.0)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_steps += 1

                step_loss = accum_loss_sum / max(accum_valid_tokens, 1)
                accum_loss_sum = 0.0
                accum_valid_tokens = 0
                pbar.set_postfix({"loss": f"{step_loss:.4f}"})

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch, "step": total_steps, "loss": step_loss}) + "\n")

    saved_path = save_checkpoint(
        model=unwrap_peft_model(model),
        processor=processor,
        tag=f"sft-{track}-final",
        checkpoint_dir=checkpoint_dir,
        extra_metadata={"track": track, "epochs": epochs},
    )
    print(f"Stage 1 SFT complete. Checkpoint saved to: {saved_path}")
    return saved_path


# ---------------------------------------------------------------------------
# Backward Compatibility Helpers
# ---------------------------------------------------------------------------

def build_sft_example(
    image: Any,
    prompt_text_content: str,
    target_trace_text: str,
    processor: Any,
    system_prompt: str | None = None,
) -> dict[str, torch.Tensor]:
    """Legacy helper: build one training example with prompt masking."""
    prompt_messages = [
        {"role": "system", "content": system_prompt or "You are an expert dental AI."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text_content},
        ]},
    ]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + target_trace_text + (getattr(processor.tokenizer, "eos_token", None) or "<|im_end|>")

    image_inputs, video_inputs = safe_process_vision_info(prompt_messages)
    prompt_enc = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    full_enc = processor(text=[full_text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    labels = full_enc["input_ids"].clone()
    prompt_len = prompt_enc["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    full_enc["labels"] = labels
    return dict(full_enc)


class TraceSFTDataset(Dataset):
    """Legacy dataset expanding verified traces."""
    def __init__(self, trace_examples: list[dict[str, Any]], images_df: Any = None) -> None:
        self.samples = trace_examples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def load_trace_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Legacy loader: load trace dataset from disk."""
    if not path or not os.path.exists(str(path)):
        return []
    if str(path).endswith(".jsonl"):
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

