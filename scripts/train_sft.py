import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import transformers
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
from qwen_vl_utils import process_vision_info

@dataclass
class ScriptArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct", metadata={"help": "The model to train."})
    dataset_path: str = field(default="data/traces/train_cot_traces.jsonl", metadata={"help": "Path to the CoT traces JSONL."})
    output_dir: str = field(default="data/models/qwen_vl_sft", metadata={"help": "Output directory for the fine-tuned model."})
    batch_size: int = field(default=1, metadata={"help": "Batch size per GPU."})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "Gradient accumulation steps."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum sequence length."})


class QwenVLDataCollator:
    """
    Custom Data Collator for Qwen2.5-VL.
    Takes raw Qwen message dictionaries, extracts images, tokenizes, and pads properly.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages_batch = [feature["messages"] for feature in features]
        
        # Apply chat template to get raw text with <|vision_start|>... tags
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages_batch
        ]
        
        # Extract images from the messages
        image_inputs_batch = []
        video_inputs_batch = []
        for msg in messages_batch:
            image_inputs, video_inputs = process_vision_info(msg)
            image_inputs_batch.append(image_inputs)
            video_inputs_batch.append(video_inputs)
            
        # The processor expects a flattened list of images if passing multiple, or nested lists.
        # process_vision_info already returns a list of images/videos per batch item.
        # We need to flatten them for the processor batch call
        flat_images = []
        for imgs in image_inputs_batch:
            if imgs is not None:
                flat_images.extend(imgs if isinstance(imgs, list) else [imgs])
                
        flat_videos = []
        for vids in video_inputs_batch:
            if vids is not None:
                flat_videos.extend(vids if isinstance(vids, list) else [vids])

        # Tokenize everything natively via Qwen's processor
        batch = self.processor(
            text=texts,
            images=flat_images if flat_images else None,
            videos=flat_videos if flat_videos else None,
            padding=True,
            return_tensors="pt"
        )
        
        # SFT needs 'labels' to calculate loss
        # For standard autoregressive LM training, labels = input_ids
        labels = batch["input_ids"].clone()
        
        # (Optional but recommended): Mask the user prompt in the labels so loss is only on the assistant's reasoning.
        # Qwen's processor padding token is usually the pad_token_id, we should mask that out too.
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        batch["labels"] = labels
        return batch


def prepare_dataset(jsonl_path: str) -> Dataset:
    """
    Parse the output from run_daily_trace_generator.py and convert to HF Dataset format.
    Expects jsonl format: {"image_path": "...", "verified_traces": [{"messages": [...]}]}
    """
    all_messages = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line.strip())
                image_path = record.get("image_path")
                
                # Check if this image generated grounded traces
                traces = record.get("verified_traces", [])
                for trace in traces:
                    messages = trace.get("messages", [])
                    if not messages: continue
                    
                    # We need to replace the "<Image>" placeholder with the actual file path
                    # Qwen expects dicts like {"type": "image", "image": "file:///path"}
                    for msg in messages:
                        if isinstance(msg.get("content"), list):
                            for content_block in msg["content"]:
                                if content_block.get("type") == "image" and content_block.get("image") == "<Image>":
                                    content_block["image"] = f"file://{image_path}"
                    
                    all_messages.append({"messages": messages})
            except Exception as e:
                print(f"Skipping malformed line: {e}")
                
    print(f"Loaded {len(all_messages)} verified multi-modal traces.")
    return Dataset.from_list(all_messages)


def main():
    parser = transformers.HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # 1. Load Dataset
    print(f"Loading dataset from {script_args.dataset_path}...")
    if not os.path.exists(script_args.dataset_path):
        print(f"Error: Dataset {script_args.dataset_path} not found.")
        return
        
    dataset = prepare_dataset(script_args.dataset_path)
    if len(dataset) == 0:
        print("Error: No valid traces found in dataset.")
        return
    
    print(f"Loading processor for {script_args.model_id}...")
    processor = AutoProcessor.from_pretrained(script_args.model_id, trust_remote_code=True)
    
    # Ensure pad token is set for the collator
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # 2. Load Model in 4-bit for QLoRA
    print("Loading model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        script_args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 3. Configure LoRA Adapters
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Configure Training
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        num_train_epochs=script_args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none", # Set to "wandb" if you use Weights & Biases
        remove_unused_columns=False, # CRITICAL FOR CUSTOM COLLATORS!
    )
    
    collator = QwenVLDataCollator(processor=processor)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=processor.tokenizer,
        args=training_args,
        data_collator=collator
    )
    
    print("Starting Vision-Language Supervised Fine-Tuning (SFT)...")
    trainer.train()
    
    # Save the final adapter
    print(f"Saving fine-tuned adapter to {script_args.output_dir}...")
    trainer.save_model(script_args.output_dir)
    processor.save_pretrained(script_args.output_dir)
    print("SFT Training Complete!")

if __name__ == "__main__":
    main()
