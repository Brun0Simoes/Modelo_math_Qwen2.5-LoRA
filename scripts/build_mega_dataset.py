"""Build mega dataset from GSM8K + MATH + our curated examples.

Downloads public math datasets from HuggingFace, converts to our
prompt/response JSONL format, and merges with curated examples.

Usage:
    cd E:\\IA_matematica
    & .\\.venv\\Scripts\\python.exe .\\scripts\\build_mega_dataset.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import random
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
for s in ("stdout", "stderr"):
    stream = getattr(sys, s, None)
    if stream:
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_math_answer(answer: str) -> str:
    """Clean up a math answer string."""
    answer = answer.strip()
    # Remove \boxed{} wrapper
    m = re.search(r"\\boxed\{(.+)\}", answer)
    if m:
        answer = m.group(1)
    return answer.strip()


def convert_gsm8k(max_examples: int = 4000) -> list[dict]:
    """Download and convert GSM8K dataset."""
    from datasets import load_dataset
    
    print("[INFO] Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"[INFO] GSM8K: {len(ds)} examples available")
    
    examples = []
    for row in ds:
        question = row["question"].strip()
        answer_text = row["answer"].strip()
        
        # GSM8K format: step-by-step reasoning ending with #### <number>
        parts = answer_text.split("####")
        if len(parts) == 2:
            reasoning = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            reasoning = answer_text
            final_answer = ""
        
        prompt = f"Resolva passo a passo.\n\n{question}"
        response = f"{reasoning}\n\nFinal answer: {final_answer}"
        
        examples.append({"prompt": prompt, "response": response})
        
        if len(examples) >= max_examples:
            break
    
    print(f"[INFO] GSM8K: converted {len(examples)} examples")
    return examples


def _try_load_dataset(names_and_configs):
    """Try multiple dataset sources, return the first that works."""
    from datasets import load_dataset
    for name, config, split in names_and_configs:
        try:
            label = f"{name}" + (f"/{config}" if config else "")
            print(f"[INFO] Trying {label}...")
            kwargs = {"split": split}
            if config:
                kwargs["name"] = config
            ds = load_dataset(name, **kwargs)
            print(f"[INFO] Loaded {label}: {len(ds)} examples")
            return ds
        except Exception as e:
            print(f"[WARN] Failed {name}: {e}")
    return None


def convert_math_dataset(max_examples: int = 3000) -> list[dict]:
    """Download and convert MATH competition dataset."""
    
    print("[INFO] Downloading MATH/competition dataset...")
    
    # Try multiple sources
    ds = _try_load_dataset([
        ("lighteval/MATH", "all", "train"),
        ("lighteval/MATH", None, "train"),
        ("competition_math", None, "train"),
        ("hendrycks/competition_math", None, "train"),
        ("math_dataset", "all", "train"),
    ])
    
    if ds is None:
        # Fallback: try NuminaMath-CoT (large, high quality competition math)
        print("[INFO] Trying NuminaMath-CoT as fallback...")
        ds = _try_load_dataset([
            ("AI-MO/NuminaMath-CoT", None, "train"),
            ("TIGER-Lab/MathInstruct", None, "train"),
        ])
    
    if ds is None:
        print("[WARN] No MATH dataset available, skipping")
        return []
    
    print(f"[INFO] MATH dataset: {len(ds)} examples available")
    
    # Detect column names
    cols = ds.column_names
    prob_col = next((c for c in ["problem", "question", "query", "instruction"] if c in cols), None)
    sol_col = next((c for c in ["solution", "answer", "response", "output"] if c in cols), None)
    
    if not prob_col or not sol_col:
        print(f"[WARN] Could not detect columns. Available: {cols}")
        return []
    
    print(f"[INFO] Using columns: problem={prob_col}, solution={sol_col}")
    
    # Categorize by level/type if available
    level_col = next((c for c in ["level", "type", "subject", "source"] if c in cols), None)
    
    examples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    for idx in indices:
        if len(examples) >= max_examples:
            break
        
        row = ds[idx]
        problem = str(row.get(prob_col, "")).strip()
        solution = str(row.get(sol_col, "")).strip()
        
        if not problem or not solution or len(problem) < 10:
            continue
        
        # Extract final answer from \boxed{}
        final_ans = clean_math_answer(solution)
        
        prompt = f"Resolva passo a passo.\n\n{problem}"
        response = f"{solution}\n\nFinal answer: {final_ans}"
        
        examples.append({"prompt": prompt, "response": response})
    
    print(f"[INFO] MATH: converted {len(examples)} examples")
    return examples


def load_curated_examples() -> list[dict]:
    """Load our curated Portuguese examples."""
    examples = []
    for fname in ["production_dataset_v2.jsonl", "competitive_enem_obmep.jsonl", "advanced_training_v2.jsonl"]:
        fpath = OUT_DIR / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
    
    # Deduplicate by prompt
    seen = set()
    unique = []
    for ex in examples:
        key = ex.get("prompt", "")[:200]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    
    print(f"[INFO] Curated PT examples: {len(unique)}")
    return unique


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-max", type=int, default=4000)
    parser.add_argument("--math-max", type=int, default=3000)
    parser.add_argument("--output", default=str(OUT_DIR / "mega_dataset_v3.jsonl"))
    args = parser.parse_args()
    
    random.seed(42)
    
    # 1. Load curated Portuguese examples (highest priority)
    curated = load_curated_examples()
    
    # 2. Download and convert GSM8K
    gsm8k = convert_gsm8k(max_examples=args.gsm8k_max)
    
    # 3. Download and convert MATH
    math_ds = convert_math_dataset(max_examples=args.math_max)
    
    # 4. Merge: curated first (they appear more in training), then public
    all_examples = curated + gsm8k + math_ds
    
    # 5. Shuffle (but keep curated examples duplicated for higher weight)
    # Add curated examples 3x for emphasis (they're our best quality)
    weighted = curated * 3 + gsm8k + math_ds
    random.shuffle(weighted)
    
    # 6. Save
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in weighted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    total = len(weighted)
    print(f"\n{'='*60}")
    print(f"  DATASET FINAL: {total} exemplos")
    print(f"  Curated PT (3x): {len(curated) * 3}")
    print(f"  GSM8K: {len(gsm8k)}")
    print(f"  MATH: {len(math_ds)}")
    print(f"  Salvo: {output_path}")
    print(f"  Tamanho: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
