import argparse
import os
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def ensure_output_on_project_drive(output_dir: str, project_root: str) -> None:
    out_abs = os.path.abspath(output_dir).lower()
    root_abs = os.path.abspath(project_root).lower()
    if not out_abs.startswith(root_abs):
        raise ValueError(
            f"output_dir must be inside project root ({project_root}). Got: {output_dir}"
        )


def infer_lora_target_modules(model: torch.nn.Module) -> List[str]:
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "c_attn",
        "c_proj",
    ]
    linear_suffixes = set()
    for name, module in model.named_modules():
        module_name = module.__class__.__name__
        if isinstance(module, torch.nn.Linear) or module_name == "Conv1D":
            linear_suffixes.add(name.split(".")[-1])
    selected = [m for m in preferred if m in linear_suffixes]
    if not selected:
        raise ValueError("Could not infer LoRA target modules for DPO model.")
    return selected


def clean_dpo_row(row: dict) -> dict:
    prompt = str(row.get("prompt") or "").strip()
    chosen = str(row.get("chosen") or "").strip()
    rejected = str(row.get("rejected") or "").strip()
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def load_dpo_dataset(
    train_file: str,
    dataset_cache_dir: str,
    max_samples: int,
) -> Dataset:
    ds = load_dataset(
        "json",
        data_files=train_file,
        split="train",
        cache_dir=dataset_cache_dir,
    )
    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    mapped = ds.map(
        clean_dpo_row,
        remove_columns=ds.column_names,
        desc="Formatting DPO rows",
    )
    mapped = mapped.filter(
        lambda x: bool(x["prompt"] and x["chosen"] and x["rejected"]),
        desc="Filtering empty DPO rows",
    )
    return mapped


def pick_precision(precision: str) -> tuple[bool, bool, torch.dtype]:
    if precision == "fp32" or not torch.cuda.is_available():
        return (False, False, torch.float32)
    if precision == "bf16":
        return (True, False, torch.bfloat16)
    if precision == "fp16":
        return (False, True, torch.float16)

    # auto
    if torch.cuda.is_bf16_supported():
        return (True, False, torch.bfloat16)
    return (False, True, torch.float16)


def pick_device_map(device_map_mode: str):
    if not torch.cuda.is_available():
        return None
    if device_map_mode == "auto":
        return "auto"
    if device_map_mode == "cuda":
        return {"": 0}
    return None


def print_dpo_preview(ds: Dataset, print_samples: int) -> None:
    if print_samples <= 0:
        return
    for idx in range(min(print_samples, len(ds))):
        row = ds[idx]
        prompt = row["prompt"][:220].replace("\n", "\\n")
        chosen = row["chosen"][:220].replace("\n", "\\n")
        rejected = row["rejected"][:220].replace("\n", "\\n")
        print(f"[pair {idx}] prompt={prompt}")
        print(f"[pair {idx}] chosen={chosen}")
        print(f"[pair {idx}] rejected={rejected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file", required=True, help="JSONL with prompt/chosen/rejected."
    )
    parser.add_argument(
        "--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica")
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help=(
            "Optional local LoRA adapter path to continue DPO from. "
            "If set, model-name is used only as fallback tokenizer source."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="E:\\IA_matematica\\outputs\\checkpoints\\dpo_lora",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default="E:\\IA_matematica\\.cache\\hf_datasets",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all rows")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=640)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto", "none"],
        default="cuda",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--print-samples", type=int, default=1)
    args = parser.parse_args()

    ensure_output_on_project_drive(args.output_dir, args.project_root)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_cache_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    set_seed(args.seed)

    ds = load_dpo_dataset(
        train_file=args.train_file,
        dataset_cache_dir=args.dataset_cache_dir,
        max_samples=args.max_samples,
    )
    if len(ds) == 0:
        raise ValueError("No valid DPO rows with prompt/chosen/rejected after filtering.")
    print(f"Loaded DPO rows: {len(ds)} from {args.train_file}")
    print_dpo_preview(ds, args.print_samples)

    tokenizer_source = args.adapter_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16, use_fp16, torch_dtype = pick_precision(args.precision)
    device_map = pick_device_map(args.device_map)
    print(
        f"Precision: bf16={use_bf16}, fp16={use_fp16}, dtype={torch_dtype}; "
        f"device_map={device_map}"
    )

    model_kwargs = {"torch_dtype": torch_dtype}
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    peft_cfg: Optional[LoraConfig] = None
    if args.adapter_path:
        if not os.path.exists(args.adapter_path):
            raise FileNotFoundError(f"adapter-path not found: {args.adapter_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.adapter_path,
            is_trainable=True,
            **model_kwargs,
        )
        print(f"Continuing DPO from adapter: {args.adapter_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
        target_modules = infer_lora_target_modules(model)
        print("LoRA target modules:", target_modules)
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
    model.config.use_cache = False

    dpo_cfg = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=[],
        reference_free=True,
        seed=args.seed,
    )

    trainer_kwargs = dict(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    if peft_cfg is not None:
        trainer_kwargs["peft_config"] = peft_cfg

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO training done. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
