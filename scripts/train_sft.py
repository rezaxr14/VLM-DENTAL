import os
import torch
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

@dataclass
class ScriptArguments:
    model_id: str = field(default="Qwen/Qwen3-VL-2B-Instruct", metadata={"help": "The model to train."})
    dataset_path: str = field(default="data/traces/synthetic_cot_traces.jsonl", metadata={"help": "Path to the CoT traces JSONL."})
    output_dir: str = field(default="data/models/qwen3_vl_sft", metadata={"help": "Output directory for the fine-tuned model."})
    batch_size: int = field(default=2, metadata={"help": "Batch size per GPU."})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})

def format_prompt(example, processor):
    """
    Format the input JSONL row into the Qwen3-VL expected chat template.
    Expects the JSONL to have 'messages' containing standard roles.
    """
    # Note: Depending on your exact trace generation format, you might need to build the messages list here.
    # We assume 'messages' is already a list of dicts: [{"role": "user", "content": [...]}, {"role": "assistant", "content": "..."}]
    if "messages" in example:
        return processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    
    # Fallback if your dataset has 'instruction' and 'response'
    messages = [
        {"role": "user", "content": example.get("instruction", "")},
        {"role": "assistant", "content": example.get("response", "")}
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def main():
    parser = transformers.HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    print(f"Loading processor for {script_args.model_id}...")
    processor = AutoProcessor.from_pretrained(script_args.model_id, trust_remote_code=True)
    
    # 1. Load Model in 4-bit for QLoRA
    print("Loading model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # We load using standard AutoModelForCausalLM but specialized classes might be needed depending on the transformers version
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        script_args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 2. Configure LoRA Adapters
    # Target attention matrices for Vision-Language models
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
    
    # 3. Load Dataset
    print(f"Loading dataset from {script_args.dataset_path}...")
    if not os.path.exists(script_args.dataset_path):
        print(f"Warning: Dataset {script_args.dataset_path} not found. Please ensure Phase 1 trace generation is complete.")
        return
        
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")
    
    # Format dataset
    # TRL's SFTTrainer accepts a formatting_func or pre-tokenized dataset
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])): # Assuming batched
            text = f"<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{example['response'][i]}<|im_end|>"
            output_texts.append(text)
        return output_texts

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
        report_to="none" # Set to "wandb" if you use Weights & Biases
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=processor.tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func # Custom formatter
    )
    
    print("Starting Supervised Fine-Tuning (SFT)...")
    trainer.train()
    
    # Save the final adapter
    print(f"Saving fine-tuned adapter to {script_args.output_dir}...")
    trainer.save_model(script_args.output_dir)
    processor.save_pretrained(script_args.output_dir)
    print("SFT Training Complete!")

if __name__ == "__main__":
    main()
