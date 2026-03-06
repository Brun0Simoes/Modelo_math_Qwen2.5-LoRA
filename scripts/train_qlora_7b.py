"""QLoRA training script for Qwen2.5-Math-7B on competitive math data.

Usage:
    cd E:\\IA_matematica
    & .\\.venv\\Scripts\\python.exe .\\scripts\\train_qlora_7b.py ^
        --train-file .\\data\\processed\\competitive_enem_obmep.jsonl ^
        --output-dir .\\outputs\\checkpoints\\qwen25math7b_qlora_competitive

Requires: bitsandbytes, peft, transformers, trl, torch (CUDA).
Fits in ~6GB VRAM on RTX 3070 8GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream:
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def format_chat(row: dict) -> dict:
    """Convert prompt/response to chat format for SFT."""
    prompt = row.get("prompt", row.get("problem", ""))
    response = row.get("response", row.get("solution", ""))
    return {
        "messages": [
            {"role": "system", "content": "Você é um assistente especialista em matemática. Resolva problemas passo a passo de forma clara e precisa. Sempre termine com 'Final answer: <resposta>'."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="QLoRA 4-bit SFT for Qwen2.5-Math-7B")
    parser.add_argument("--train-file", required=True, help="Path to JSONL training data")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path to checkpoint dir to resume training from")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] VRAM: {total_vram:.1f} GB")

    # ── Load data ──
    raw_data = load_dataset(args.train_file)
    print(f"[INFO] Loaded {len(raw_data)} training examples from {args.train_file}")
    chat_data = [format_chat(row) for row in raw_data]

    # ── 4-bit quantization config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model in 4-bit ──
    print(f"[INFO] Loading {args.model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Report VRAM after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"[INFO] VRAM after model load: {allocated:.2f} GB")

    # ── LoRA config ──
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[INFO] Trainable params: {trainable:,d} / {total:,d} ({100 * trainable / total:.2f}%)")

    # ── Training ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_length=args.max_seq_len,
        report_to="none",
    )

    # Format dataset for SFTTrainer
    from datasets import Dataset

    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = Dataset.from_list(chat_data)
    ds = ds.map(apply_chat_template, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=training_args,
    )

    print(f"[INFO] Starting training: {args.epochs} epochs, effective batch size = {args.batch_size * args.grad_accum}")
    if args.resume_from_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save adapter ──
    print(f"[INFO] Saving adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training config for reference
    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps({
        "model_name": args.model_name,
        "train_file": args.train_file,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "quantization": "4-bit NF4 double-quant",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] Training complete!")
    print(f"[INFO] Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
