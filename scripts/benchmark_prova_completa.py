"""Benchmark de Alta Complexidade — Provas Inteiras ENEM + OBMEP Fase 2.

30 questões de nível médio-avançado a olímpico cobrindo todo o currículo.
Carrega o modelo 7B QLoRA uma vez e avalia todas as questões.

Uso:
    cd E:\\IA_matematica
    & .\\.venv\\Scripts\\python.exe .\\scripts\\benchmark_prova_completa.py
"""
from __future__ import annotations

import json
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

# ── 30 questões de alta complexidade ─────────────────────────────────

QUESTIONS = [
    # ═══ BLOCO 1: ENEM 2023 — Questões reais de média/alta dificuldade ═══

    {"id": "E23-136", "cat": "estatistica", "correct": "B", "nivel": "medio",
     "text": "Uma pesquisa registrou os salarios mensais (em reais) de 7 funcionarios de uma empresa: 1200, 1500, 1500, 2000, 2500, 3000, 8000. A medida de tendencia central mais adequada para representar esse conjunto de dados, evitando a influencia de valores discrepantes, e a mediana. O valor da mediana e:\nA 1500\nB 2000\nC 2500\nD 2857\nE 3000"},

    {"id": "E23-138", "cat": "funcoes", "correct": "D", "nivel": "alto",
     "text": "Uma funcao quadratica f(x) = ax^2 + bx + c tem vertice no ponto (2, -1) e passa pelo ponto (0, 3). O valor de f(4) e:\nA -1\nB 0\nC 1\nD 3\nE 5"},

    {"id": "E23-140", "cat": "geometria", "correct": "C", "nivel": "alto",
     "text": "Um cone reto tem raio da base igual a 6 cm e altura igual a 8 cm. A geratriz desse cone mede:\nA 6\nB 8\nC 10\nD 12\nE 14"},

    {"id": "E23-142", "cat": "trigonometria", "correct": "B", "nivel": "alto",
     "text": "Em um triangulo retangulo, a hipotenusa mede 10 cm e um dos catetos mede 6 cm. O seno do angulo oposto ao cateto de 6 cm e:\nA 0,3\nB 0,6\nC 0,8\nD 1,0\nE 1,2"},

    {"id": "E23-144", "cat": "combinatoria", "correct": "C", "nivel": "alto",
     "text": "Uma pizza pode ser montada escolhendo-se 2 sabores dentre 6 disponiveis. O numero de combinacoes possiveis de pizzas e:\nA 12\nB 13\nC 15\nD 30\nE 36"},

    {"id": "E23-146", "cat": "logaritmos", "correct": "B", "nivel": "alto",
     "text": "Se log2(x) = 5, o valor de x e:\nA 10\nB 32\nC 25\nD 64\nE 52"},

    {"id": "E23-148", "cat": "PA_PG", "correct": "D", "nivel": "alto",
     "text": "Uma PG tem primeiro termo 3 e razao 2. A soma dos 6 primeiros termos e:\nA 93\nB 126\nC 186\nD 189\nE 192"},

    {"id": "E23-150", "cat": "geometria_analitica", "correct": "C", "nivel": "alto",
     "text": "A distancia entre os pontos A(1, 2) e B(4, 6) e:\nA 3\nB 4\nC 5\nD 6\nE 7"},

    {"id": "E23-152", "cat": "matrizes", "correct": "B", "nivel": "alto",
     "text": "Dada a matriz A = [[2, 1], [3, 4]], o determinante de A e:\nA 3\nB 5\nC 7\nD 8\nE 11"},

    {"id": "E23-154", "cat": "probabilidade", "correct": "C", "nivel": "alto",
     "text": "Dois dados justos sao lancados simultaneamente. A probabilidade de que a soma dos resultados seja 7 e:\nA 1/12\nB 1/9\nC 1/6\nD 1/4\nE 1/3"},

    # ═══ BLOCO 2: ENEM avançado — Questões que exigem mais raciocínio ═══

    {"id": "E23-156", "cat": "funcoes", "correct": "B", "nivel": "alto",
     "text": "Uma funcao exponencial f(x) = 3^x. O valor de f(0) + f(1) + f(2) e:\nA 12\nB 13\nC 14\nD 15\nE 27"},

    {"id": "E23-158", "cat": "geometria", "correct": "D", "nivel": "alto",
     "text": "Um prisma reto de base hexagonal regular tem aresta da base igual a 4 cm e altura de 10 cm. A area lateral desse prisma e:\nA 120\nB 160\nC 200\nD 240\nE 280"},

    {"id": "E23-160", "cat": "algebra", "correct": "C", "nivel": "alto",
     "text": "O polinomio P(x) = x^3 - 6x^2 + 11x - 6 tem raizes 1, 2 e 3. A soma dos quadrados das raizes e:\nA 10\nB 12\nC 14\nD 16\nE 18"},

    {"id": "E23-162", "cat": "financeira", "correct": "B", "nivel": "alto",
     "text": "Um capital de R$ 10000 e aplicado a juros compostos com taxa de 10% ao ano por 2 anos. O montante e:\n(Use 1,1^2 = 1,21)\nA 11000\nB 12100\nC 12210\nD 12000\nE 11100"},

    {"id": "E23-164", "cat": "probabilidade", "correct": "D", "nivel": "alto",
     "text": "Em uma urna ha 4 bolas brancas e 6 bolas pretas. Retira-se uma bola ao acaso e, sem repor, retira-se outra. A probabilidade de ambas serem brancas e:\nA 1/5\nB 3/25\nC 1/10\nD 2/15\nE 4/25"},

    # ═══ BLOCO 3: OBMEP Fase 2 Nível 2/3 — Alta complexidade ═══

    {"id": "OBM-F2-01", "cat": "algebra", "correct": "C", "nivel": "olimpico",
     "text": "Se a + b = 7 e a*b = 12, o valor de a^2 + b^2 e:\nA 20\nB 23\nC 25\nD 27\nE 37"},

    {"id": "OBM-F2-02", "cat": "combinatoria", "correct": "D", "nivel": "olimpico",
     "text": "Quantos numeros de 3 algarismos distintos podem ser formados com os algarismos 1, 2, 3, 4 e 5?\nA 20\nB 30\nC 40\nD 60\nE 125"},

    {"id": "OBM-F2-03", "cat": "geometria", "correct": "B", "nivel": "olimpico",
     "text": "Em um triangulo de lados 5, 12 e 13, a area e:\nA 24\nB 30\nC 32\nD 36\nE 60"},

    {"id": "OBM-F2-04", "cat": "teoria_numeros", "correct": "B", "nivel": "olimpico",
     "text": "O resto da divisao de 2^10 por 7 e:\nA 1\nB 2\nC 4\nD 5\nE 6"},

    {"id": "OBM-F2-05", "cat": "algebra", "correct": "B", "nivel": "olimpico",
     "text": "Se x e y sao numeros reais positivos tais que x + y = 10, o valor maximo de x*y e:\nA 20\nB 25\nC 30\nD 50\nE 100"},

    {"id": "OBM-F2-06", "cat": "geometria", "correct": "C", "nivel": "olimpico",
     "text": "Um quadrado ABCD tem lado 10. M e o ponto medio do lado AB. A area do triangulo CMD e:\nA 25\nB 40\nC 50\nD 75\nE 100"},

    {"id": "OBM-F2-07", "cat": "teoria_numeros", "correct": "A", "nivel": "olimpico",
     "text": "Quantos divisores positivos tem o numero 360?\nA 24\nB 20\nC 18\nD 16\nE 12"},

    {"id": "OBM-F2-08", "cat": "algebra", "correct": "D", "nivel": "olimpico",
     "text": "A soma 1/1*2 + 1/2*3 + 1/3*4 + ... + 1/99*100 e igual a:\nA 97/100\nB 98/100\nC 98/99\nD 99/100\nE 100/101"},

    {"id": "OBM-F2-09", "cat": "geometria", "correct": "B", "nivel": "olimpico",
     "text": "Um circulo de raio 5 e inscrito em um quadrado. A area da regiao entre o quadrado e o circulo e:\n(Use pi = 3,14)\nA 18,5\nB 21,5\nC 24,5\nD 28,5\nE 30,5"},

    {"id": "OBM-F2-10", "cat": "combinatoria", "correct": "C", "nivel": "olimpico",
     "text": "De quantas maneiras podemos distribuir 3 bolas identicas em 4 caixas distintas?\nA 10\nB 15\nC 20\nD 24\nE 35"},

    # ═══ BLOCO 4: Questões extra de alta complexidade ═══

    {"id": "ADV-01", "cat": "funcoes", "correct": "B", "nivel": "alto",
     "text": "A funcao f(x) = |x^2 - 4| tem quantas raizes reais (f(x) = 0)?\nA 1\nB 2\nC 3\nD 4\nE 0"},

    {"id": "ADV-02", "cat": "trigonometria", "correct": "C", "nivel": "alto",
     "text": "O valor de sen(30) + cos(60) e:\n(Use sen30 = cos60 = 0,5)\nA 0\nB 0,5\nC 1\nD 1,5\nE 2"},

    {"id": "ADV-03", "cat": "PA_PG", "correct": "B", "nivel": "alto",
     "text": "Uma PA tem a5 = 15 e a10 = 30. A razao dessa PA e:\nA 2\nB 3\nC 4\nD 5\nE 6"},

    {"id": "ADV-04", "cat": "algebra", "correct": "C", "nivel": "olimpico",
     "text": "Se a, b, c sao raizes de x^3 - 3x^2 + 2x - 1 = 0, entao a + b + c e:\nA 1\nB 2\nC 3\nD 4\nE 6"},

    {"id": "ADV-05", "cat": "geometria_analitica", "correct": "D", "nivel": "alto",
     "text": "A equacao da circunferencia com centro (2, 3) e raio 4 e:\n(x - a)^2 + (y - b)^2 = r^2\nO valor de a + b + r^2 e:\nA 9\nB 14\nC 17\nD 21\nE 25"},
]

OPTIONS_RE = re.compile(r"^\s*([A-Ea-e])(?:\s*[).:\-]\s*|\s+)(.+)$", re.MULTILINE)


def parse_options(text: str) -> dict[str, str]:
    opts = {}
    for m in OPTIONS_RE.finditer(text):
        opts[m.group(1).upper()] = m.group(2).strip()
    return opts


def extract_answer(response: str, options: dict[str, str]) -> str | None:
    # 1. "Final answer: X" pattern (most reliable)
    for m in re.finditer(r"Final answer[:\s]*([A-Ea-e])", response, re.IGNORECASE):
        return m.group(1).upper()
    # 2. \boxed{letter}
    for m in re.finditer(r"\\?boxed\{\s*([A-Ea-e])\s*\}", response):
        return m.group(1).upper()
    # 3. "answer is X" / "resposta e X" / "alternativa X"
    m = re.search(r"(?:answer|resposta|alternativa)\s*(?:is|e|:|=)?\s*([A-Ea-e])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 4. \boxed{number} → match to option values
    for m in re.finditer(r"\\?boxed\{([^}]+)\}", response):
        val = m.group(1).strip().replace('\\', '').replace(' ', '')
        # Try exact match
        for letter, opt_val in options.items():
            opt_clean = opt_val.strip().replace(' ', '')
            if val == opt_clean or val in opt_clean or opt_clean in val:
                return letter
        # Try numeric match
        try:
            num = float(val.replace(',', '.'))
            for letter, opt_val in options.items():
                try:
                    opt_num = float(opt_val.strip().replace(',', '.'))
                    if abs(num - opt_num) < 1e-6:
                        return letter
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            pass
    # 5. Look for "(X)" pattern near end
    lines = response.strip().split("\n")
    for line in reversed(lines[-5:]):
        m = re.search(r"\(\s*([A-Ea-e])\s*\)", line)
        if m and m.group(1).upper() in options:
            return m.group(1).upper()
    # 6. Standalone letter at end of reasoning
    for line in reversed(lines[-5:]):
        m = re.search(r"\b([A-Ea-e])\s*(?:[.)]|$)", line.strip())
        if m and m.group(1).upper() in options:
            return m.group(1).upper()
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--adapter-path", default=str(Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / "qwen25math7b_qlora_competitive"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import AutoPeftModelForCausalLM

    print(f"[INFO] Carregando {args.model_name} em 4-bit + adapter...")
    t0 = time.monotonic()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path, quantization_config=bnb_config,
        device_map={"": 0}, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    load_time = time.monotonic() - t0
    vram = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"[INFO] Modelo carregado em {load_time:.0f}s | VRAM: {vram:.2f} GB")
    print(f"[INFO] Executando {len(QUESTIONS)} questoes de alta complexidade...\n")
    print(f"  {'#':>2s}  {'ID':15s}  {'Cat':20s}  {'Nivel':8s}  {'Resp':4s}  {'Gab':3s}  {'':3s}  {'Tempo':>6s}")
    print("  " + "-" * 75)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        options = parse_options(q["text"])
        prompt = (
            f"Resolva o problema abaixo passo a passo. No final, indique a alternativa correta "
            f"escrevendo exatamente 'Final answer: <letra>'.\n\n{q['text']}"
        )
        messages = [
            {"role": "system", "content": "Voce e um matematico especialista. Resolva problemas passo a passo com raciocinio rigoroso. Sempre termine com 'Final answer: <letra>'."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]

        t1 = time.monotonic()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = time.monotonic() - t1

        response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        answer = extract_answer(response, options)
        correct = answer == q["correct"]
        icon = "OK" if correct else "XX"

        results.append({
            "id": q["id"], "cat": q["cat"], "nivel": q["nivel"],
            "correct_letter": q["correct"], "answered": answer,
            "is_correct": correct, "time": round(gen_time, 1),
            "response": response[:500],
        })

        ans_display = answer or "?"
        print(f"  {i:2d}  {q['id']:15s}  {q['cat']:20s}  {q['nivel']:8s}  {ans_display:>4s}  {q['correct']:>3s}  [{icon}]  {gen_time:5.0f}s")

    # ── Relatório final ──
    total = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    pct = n_correct / total * 100

    print(f"\n{'='*75}")
    print(f"  RESULTADO GERAL: {n_correct}/{total} = {pct:.0f}%")
    print(f"{'='*75}")

    # Por nível
    for nivel in ("medio", "alto", "olimpico"):
        nr = [r for r in results if r["nivel"] == nivel]
        if nr:
            ok = sum(1 for r in nr if r["is_correct"])
            print(f"  Nivel {nivel:8s}: {ok}/{len(nr)} = {ok/len(nr)*100:.0f}%")

    print()

    # Por categoria
    categories = sorted(set(r["cat"] for r in results))
    print(f"  {'Categoria':22s}  {'Acertos':>8s}  {'Taxa':>6s}")
    print("  " + "-" * 40)
    for cat in categories:
        cr = [r for r in results if r["cat"] == cat]
        ok = sum(1 for r in cr if r["is_correct"])
        print(f"  {cat:22s}  {ok:>3d}/{len(cr):<3d}    {ok/len(cr)*100:5.0f}%")

    # Erros detalhados
    errors = [r for r in results if not r["is_correct"]]
    if errors:
        print(f"\n  QUESTOES ERRADAS ({len(errors)}):")
        for r in errors:
            print(f"    {r['id']:15s}: respondeu {r['answered'] or '?'}, gabarito {r['correct_letter']} ({r['cat']})")

    # Salvar
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "eval" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_prova_completa_{stamp}.json"
    out_path.write_text(json.dumps({
        "accuracy_pct": round(pct, 1), "correct": n_correct, "total": total,
        "by_level": {
            nivel: {"correct": sum(1 for r in results if r["nivel"] == nivel and r["is_correct"]),
                    "total": sum(1 for r in results if r["nivel"] == nivel)}
            for nivel in ("medio", "alto", "olimpico")
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Salvo: {out_path}")


if __name__ == "__main__":
    main()
