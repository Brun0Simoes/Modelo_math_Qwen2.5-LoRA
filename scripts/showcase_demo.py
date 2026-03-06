"""Showcase: Run BEST model on genuinely complex math problems.

Loads the BEST QLoRA adapter, solves 15 hard problems (ITA/OBMEP/Unicamp level),
and outputs beautifully formatted reasoning chains for GitHub showcase.

Usage:
    cd E:\\IA_matematica
    & .\\.venv\\Scripts\\python.exe .\\scripts\\showcase_demo.py
"""
from __future__ import annotations
import json, os, re, sys, time
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

# ── 15 genuinely complex problems (ITA, OBMEP, Unicamp, competition level) ──
SHOWCASE_PROBLEMS = [
    {
        "id": "ITA-01",
        "title": "ITA 2023 - Algebra Avancada",
        "category": "Algebra",
        "difficulty": "ITA/IME",
        "problem": (
            "Se x e y sao numeros reais positivos tais que x + y = 10 e x*y = 21, "
            "determine o valor de x^3 + y^3."
        ),
        "expected": "370",
    },
    {
        "id": "OBMEP-F3-01",
        "title": "OBMEP Fase 3 - Teoria dos Numeros",
        "category": "Teoria dos Numeros",
        "difficulty": "Olimpico",
        "problem": (
            "Encontre o resto da divisao de 7^2023 por 11."
        ),
        "expected": "2",
    },
    {
        "id": "UNICAMP-01",
        "title": "Unicamp 2023 - Logaritmos",
        "category": "Logaritmos",
        "difficulty": "Vestibular Dificil",
        "problem": (
            "Se log_2(x) + log_4(x) + log_8(x) = 11, determine o valor de x."
        ),
        "expected": "64",
    },
    {
        "id": "ITA-02",
        "title": "ITA - Numeros Complexos",
        "category": "Numeros Complexos",
        "difficulty": "ITA/IME",
        "problem": (
            "Calcule o modulo do numero complexo z = (2 + 3i)^2 + (1 - i)^3. "
            "Expresse o resultado na forma simplificada."
        ),
        "expected": "",
    },
    {
        "id": "OBMEP-F3-02",
        "title": "OBMEP Fase 3 - Combinatoria Avancada",
        "category": "Combinatoria",
        "difficulty": "Olimpico",
        "problem": (
            "De quantas maneiras podemos pintar os vertices de um hexagono regular "
            "com 3 cores disponiveis, de modo que vertices adjacentes tenham cores "
            "diferentes? (Considere rotacoes como pinturas distintas)"
        ),
        "expected": "246",
    },
    {
        "id": "FUVEST-01",
        "title": "Fuvest 2023 - Funcoes",
        "category": "Funcoes",
        "difficulty": "Vestibular Dificil",
        "problem": (
            "Seja f(x) = x^2 - 4x + 3. Determine todos os valores de x para os quais "
            "f(f(x)) = 3."
        ),
        "expected": "",
    },
    {
        "id": "ITA-03",
        "title": "ITA - Inequacoes com Modulo",
        "category": "Algebra",
        "difficulty": "ITA/IME",
        "problem": (
            "Resolva a inequacao |x^2 - 5x + 4| < 2. "
            "Apresente o conjunto solucao."
        ),
        "expected": "",
    },
    {
        "id": "OBMEP-F3-03",
        "title": "OBMEP Fase 3 - Geometria",
        "category": "Geometria",
        "difficulty": "Olimpico",
        "problem": (
            "Em um triangulo ABC, o angulo A mede 60 graus, AB = 8 e AC = 5. "
            "Calcule o comprimento de BC usando a lei dos cossenos."
        ),
        "expected": "7",
    },
    {
        "id": "UNICAMP-02",
        "title": "Unicamp - Sequencias",
        "category": "PA/PG",
        "difficulty": "Vestibular Dificil",
        "problem": (
            "A sequencia definida por a_1 = 1 e a_(n+1) = 2*a_n + 1 para n >= 1. "
            "Encontre uma formula fechada para a_n e calcule a_10."
        ),
        "expected": "1023",
    },
    {
        "id": "ITA-04",
        "title": "ITA - Probabilidade Condicional",
        "category": "Probabilidade",
        "difficulty": "ITA/IME",
        "problem": (
            "Uma urna contem 5 bolas brancas e 3 pretas. Retira-se 2 bolas sem reposicao. "
            "Sabendo que pelo menos uma das bolas retiradas e branca, qual a probabilidade "
            "de ambas serem brancas?"
        ),
        "expected": "10/25",
    },
    {
        "id": "OBMEP-F3-04",
        "title": "OBMEP Fase 3 - Polinomios",
        "category": "Algebra",
        "difficulty": "Olimpico",
        "problem": (
            "Seja P(x) = x^4 - 4x^3 + 6x^2 - 4x + 1. "
            "Fatore completamente P(x) e encontre todas as suas raizes."
        ),
        "expected": "x=1 (multiplicidade 4)",
    },
    {
        "id": "UNICAMP-03",
        "title": "Unicamp - Geometria Analitica",
        "category": "Geometria Analitica",
        "difficulty": "Vestibular Dificil",
        "problem": (
            "Determine a equacao da circunferencia que passa pelos pontos A(0,0), "
            "B(4,0) e C(0,6). Encontre o centro e o raio."
        ),
        "expected": "centro (2,3), raio sqrt(13)",
    },
    {
        "id": "IME-01",
        "title": "IME - Calculo Diferencial",
        "category": "Calculo",
        "difficulty": "ITA/IME",
        "problem": (
            "Determine o valor maximo e minimo da funcao f(x) = x^3 - 3x^2 + 4 "
            "no intervalo [-1, 3]."
        ),
        "expected": "max=4 em x=0, min=0 em x=2",
    },
    {
        "id": "OBMEP-F3-05",
        "title": "OBMEP Fase 3 - Congruencia Modular",
        "category": "Teoria dos Numeros",
        "difficulty": "Olimpico",
        "problem": (
            "Encontre o ultimo algarismo (unidade) de 3^2024 + 7^2024."
        ),
        "expected": "2",
    },
    {
        "id": "ITA-05",
        "title": "ITA - Matrizes e Determinantes",
        "category": "Matrizes",
        "difficulty": "ITA/IME",
        "problem": (
            "Seja A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]. "
            "Calcule o determinante de A."
        ),
        "expected": "1",
    },
]


def load_model(adapter_path: str):
    """Load the BEST model with 4-bit quantization."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import AutoPeftModelForCausalLM

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def solve(model, tokenizer, problem_text: str, max_tokens: int = 1024) -> str:
    """Solve a problem with the model."""
    import torch

    messages = [
        {"role": "system", "content": "Voce e um especialista em matematica. Resolva problemas passo a passo de forma clara, mostrando todo o raciocinio. Sempre termine com 'Final answer: <resposta>'."},
        {"role": "user", "content": f"Resolva passo a passo.\n\n{problem_text}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            temperature=1.0, top_p=1.0, repetition_penalty=1.1,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=str(ROOT / "outputs" / "checkpoints" / "qwen25math7b_BEST"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "showcase"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SHOWCASE: Qwen2.5-Math-7B + QLoRA Fine-tuned")
    print("  Resolvendo problemas nivel ITA/OBMEP/Unicamp")
    print("=" * 70)
    print()

    # Load model
    print("[INFO] Carregando modelo BEST...")
    t0 = time.monotonic()
    model, tokenizer = load_model(args.adapter_path)
    load_time = time.monotonic() - t0

    import torch
    vram = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"[INFO] Modelo carregado em {load_time:.0f}s | VRAM: {vram:.2f} GB\n")

    results = []
    for i, prob in enumerate(SHOWCASE_PROBLEMS, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(SHOWCASE_PROBLEMS)}] {prob['title']}")
        print(f"  Dificuldade: {prob['difficulty']} | Categoria: {prob['category']}")
        print(f"{'='*70}")
        print(f"\n  PROBLEMA:")
        print(f"  {prob['problem']}\n")

        t1 = time.monotonic()
        response = solve(model, tokenizer, prob["problem"], max_tokens=args.max_new_tokens)
        elapsed = time.monotonic() - t1

        print(f"  RESOLUCAO DO MODELO ({elapsed:.0f}s):")
        print(f"  {'-'*60}")
        for line in response.split("\n"):
            print(f"  {line}")
        print(f"  {'-'*60}")

        if prob["expected"]:
            correct = prob["expected"].lower() in response.lower()
            status = "CORRETO" if correct else "VERIFICAR"
            print(f"\n  Esperado: {prob['expected']}")
            print(f"  Status: [{status}]")
        else:
            status = "ABERTO"
            print(f"\n  Status: [RESPOSTA ABERTA - verificar manualmente]")

        results.append({
            "id": prob["id"],
            "title": prob["title"],
            "category": prob["category"],
            "difficulty": prob["difficulty"],
            "problem": prob["problem"],
            "expected": prob["expected"],
            "model_response": response,
            "time_seconds": round(elapsed, 1),
            "status": status,
        })

    # Save results
    results_path = out_dir / "showcase_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    correct = sum(1 for r in results if r["status"] == "CORRETO")
    verify = sum(1 for r in results if r["status"] == "VERIFICAR")
    open_ans = sum(1 for r in results if r["status"] == "ABERTO")
    total_time = sum(r["time_seconds"] for r in results)

    print(f"\n\n{'='*70}")
    print(f"  RESUMO SHOWCASE")
    print(f"{'='*70}")
    print(f"  Corretos: {correct}/{len(results)}")
    print(f"  Para verificar: {verify}")
    print(f"  Respostas abertas: {open_ans}")
    print(f"  Tempo total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Media por questao: {total_time/len(results):.0f}s")
    print(f"  Resultados: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
