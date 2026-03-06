"""Benchmark direto para modelo 7B QLoRA — carrega modelo uma vez e avalia todas as questões.

Uso:
    cd E:\\IA_matematica
    & .\\.venv\\Scripts\\python.exe .\\scripts\\benchmark_7b_direct.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

for s in ("stdout", "stderr"):
    stream = getattr(sys, s, None)
    if stream:
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ── Questions ──────────────────────────────────────────────────────────
QUESTIONS = [
    {"id": "ENEM-STAT-02", "cat": "estatistica", "correct": "C",
     "text": "Um professor registrou as notas de cinco alunos em uma prova: 6, 7, 8, 9, 10.\nA media aritmetica dessas notas e:\nA 7\nB 7,5\nC 8\nD 8,5\nE 9"},
    {"id": "ENEM-PORC-01", "cat": "aritmetica", "correct": "C",
     "text": "Um produto que custava R$ 250,00 sofreu um desconto de 20%.\nO novo preco do produto e:\nA 180\nB 190\nC 200\nD 210\nE 230"},
    {"id": "ENEM-GEO-01", "cat": "geometria", "correct": "D",
     "text": "Um terreno retangular tem 30 m de comprimento e 20 m de largura.\nA area desse terreno, em metros quadrados, e:\nA 50\nB 100\nC 500\nD 600\nE 1200"},
    {"id": "ENEM-ALG-01", "cat": "algebra", "correct": "C",
     "text": "Uma empresa cobra uma taxa fixa de R$ 50,00 mais R$ 3,00 por quilometro rodado em um servico de transporte. Se um cliente percorreu 40 km, o valor total pago foi de:\nA 120\nB 150\nC 170\nD 190\nE 200"},
    {"id": "ENEM-PROP-01", "cat": "aritmetica", "correct": "C",
     "text": "Uma torneira enche um tanque de 500 litros em 4 horas.\nMantendo a mesma vazao, em quantas horas essa torneira encheria um tanque de 1250 litros?\nA 6\nB 8\nC 10\nD 12\nE 15"},
    {"id": "ENEM-PA-01", "cat": "algebra", "correct": "C",
     "text": "Em uma progressao aritmetica, o primeiro termo e 3 e a razao e 5.\nO decimo termo dessa progressao e:\nA 43\nB 45\nC 48\nD 50\nE 53"},
    {"id": "OBMEP-N1-01", "cat": "aritmetica", "correct": "B",
     "text": "A soma de todos os numeros de dois algarismos cujos algarismos somam 5 e:\n(Exemplos: 14, pois 1+4=5; 23, pois 2+3=5; etc.)\nA 150\nB 165\nC 175\nD 195\nE 200"},
    {"id": "OBMEP-N1-02", "cat": "aritmetica", "correct": "C",
     "text": "Maria tem uma colecao de 120 figurinhas. Ela deu 1/4 da colecao para Joao e 1/3 do que sobrou para Ana.\nQuantas figurinhas Maria ainda tem?\nA 40\nB 50\nC 60\nD 70\nE 80"},
    {"id": "ENEM-PROB-01", "cat": "probabilidade", "correct": "C",
     "text": "Uma urna contem 3 bolas vermelhas, 5 bolas azuis e 2 bolas verdes. Retirando-se uma bola ao acaso, a probabilidade de que ela seja azul e:\nA 1/5\nB 3/10\nC 1/2\nD 2/5\nE 3/5"},
    {"id": "ENEM-JUR-01", "cat": "algebra", "correct": "D",
     "text": "Um capital de R$ 2000,00 foi aplicado a juros simples a taxa de 5% ao mes durante 6 meses. O montante ao final do periodo e:\nA 2300\nB 2400\nC 2500\nD 2600\nE 2800"},
    {"id": "ENEM-EQ-01", "cat": "algebra", "correct": "C",
     "text": "Um pai tem o triplo da idade de seu filho. Juntos, eles somam 52 anos. A idade do filho e:\nA 11\nB 12\nC 13\nD 14\nE 15"},
    {"id": "ENEM-GEO-02", "cat": "geometria", "correct": "D",
     "text": "Uma caixa d'agua tem formato de paralelepipedo com dimensoes internas de 2 m de comprimento, 1,5 m de largura e 1 m de altura. A capacidade dessa caixa, em litros, e:\nA 300\nB 1500\nC 2000\nD 3000\nE 4500"},
    {"id": "OBMEP-N2-01", "cat": "algebra", "correct": "D",
     "text": "Se x^2 - 7x + 12 = 0, a soma das raizes dessa equacao e:\nA 3\nB 4\nC 5\nD 7\nE 12"},
]

OPTIONS_RE = re.compile(r"^\s*([A-Ea-e])(?:\s*[).:\-]\s*|\s+)(.+)$", re.MULTILINE)


def parse_options(text: str) -> dict[str, str]:
    opts = {}
    for m in OPTIONS_RE.finditer(text):
        opts[m.group(1).upper()] = m.group(2).strip()
    return opts


def extract_answer(response: str, options: dict[str, str]) -> str | None:
    """Extract answer letter from model response."""
    # Look for "Final answer: X" pattern
    m = re.search(r"Final answer[:\s]*([A-Ea-e])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Look for \\boxed{X} or boxed{X}
    m = re.search(r"\\?boxed\{([A-Ea-e])\}", response)
    if m:
        return m.group(1).upper()
    # Look for "answer is X" 
    m = re.search(r"answer\s+is\s+([A-Ea-e])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Look for boxed number and match to options
    m = re.search(r"\\?boxed\{([^}]+)\}", response)
    if m:
        val = m.group(1).strip()
        for letter, opt_val in options.items():
            if val in opt_val or opt_val in val:
                return letter
    # Look for standalone letter at end
    m = re.search(r"\b([A-Ea-e])\s*(?:\(|$)", response.strip().split("\n")[-1])
    if m:
        return m.group(1).upper()
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--adapter-path", default=str(Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / "qwen25math7b_qlora_competitive"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import AutoPeftModelForCausalLM

    print(f"[INFO] Loading {args.model_name} in 4-bit with adapter {args.adapter_path}...")
    t0 = time.monotonic()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    load_time = time.monotonic() - t0
    vram = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"[INFO] Model loaded in {load_time:.0f}s | VRAM: {vram:.2f} GB")
    print(f"[INFO] Running {len(QUESTIONS)} questions...\n")

    # ── Run each question ──
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        options = parse_options(q["text"])
        prompt = f"Resolva passo a passo e termine com 'Final answer: <letra>'.\n\n{q['text']}"

        messages = [
            {"role": "system", "content": "Voce e um assistente especialista em matematica. Resolva problemas passo a passo. Sempre termine com 'Final answer: <letra da alternativa>'."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]

        t1 = time.monotonic()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = time.monotonic() - t1

        response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        answer = extract_answer(response, options)
        correct = answer == q["correct"]
        icon = "OK" if correct else "XX"

        results.append({
            "id": q["id"],
            "cat": q["cat"],
            "correct_letter": q["correct"],
            "answered": answer,
            "is_correct": correct,
            "time": gen_time,
            "response_preview": response[:200],
        })

        print(f"  [{icon}] {q['id']:15s} | {answer or '?':1s} (esperado {q['correct']}) | {gen_time:.1f}s")

    # ── Summary ──
    total = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    pct = n_correct / total * 100

    print(f"\n{'='*60}")
    print(f"  RESULTADO: {n_correct}/{total} = {pct:.0f}%")
    print(f"{'='*60}")

    categories = sorted(set(r["cat"] for r in results))
    for cat in categories:
        cat_r = [r for r in results if r["cat"] == cat]
        cat_ok = sum(1 for r in cat_r if r["is_correct"])
        print(f"  {cat:15s}: {cat_ok}/{len(cat_r)}")

    # Save
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "eval" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_7b_qlora_{stamp}.json"
    out_path.write_text(json.dumps({"accuracy_pct": round(pct, 1), "correct": n_correct, "total": total, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Salvo: {out_path}")


if __name__ == "__main__":
    main()
