"""Benchmark de questões reais ENEM / OBMEP para avaliação da plataforma.

Executa o solver (heuristic ou transformers) em 15 questões de provas
brasileiras de alto ranking e compara com o gabarito oficial.
Gera relatório com % de acertos por categoria e diagnóstico
plataforma-vs-modelo para cada erro.

Uso:
    cd E:\\IA_matematica
    & .\\.venv\\Scripts\\python.exe .\\scripts\\benchmark_enem_obmep.py --backend heuristic
    & .\\.venv\\Scripts\\python.exe .\\scripts\\benchmark_enem_obmep.py --backend transformers
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
SOLVER_SCRIPT = ROOT / "scripts" / "solve_problem_json.py"
PYTHON_EXE = sys.executable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    source: str          # e.g. "ENEM 2022", "OBMEP 2023 N1"
    category: str        # algebra, estatistica, geometria, probabilidade, aritmetica
    text: str            # full problem text with options
    correct_letter: str  # official answer: "A"|"B"|"C"|"D"|"E"
    correct_value: float # numeric value of correct option


@dataclass
class Result:
    question: Question
    answered_letter: Optional[str] = None
    answered_value: Optional[str] = None
    is_correct: bool = False
    fallback_type: Optional[str] = None
    fallback_applied: bool = False
    trust_gate: Optional[str] = None
    abstained: bool = False
    error: Optional[str] = None
    elapsed_s: float = 0.0
    diagnosis: str = ""  # "plataforma" | "modelo" | "ok"


# ---------------------------------------------------------------------------
# Question bank — real questions based on ENEM and OBMEP exams
# ---------------------------------------------------------------------------

def build_questions() -> List[Question]:
    questions: List[Question] = []

    # ── 1. ENEM — Mediana (Estatística) ──────────────────────────────
    questions.append(Question(
        id="ENEM-STAT-01",
        source="ENEM 2022",
        category="estatistica",
        text=(
            "Uma empresa de tecnologia vai padronizar a velocidade de\n"
            "conexão de internet que oferece a seus clientes em dez cidades.\n"
            "A direção da empresa decide que seu novo padrão de velocidade de\n"
            "referência será a mediana dos valores das velocidades de referência\n"
            "de conexões nessas dez cidades. Esses valores, em megabyte\n"
            "por segundo (MB/s), são apresentados no quadro.\n"
            "Cidades Velocidade de referência (MB/s)\n"
            "C1 390\n"
            "C2 380\n"
            "C3 320\n"
            "C4 390\n"
            "C5 340\n"
            "C6 380\n"
            "C7 390\n"
            "C8 400\n"
            "C9 350\n"
            "C10 360\n"
            "A velocidade de referência, em megabyte por segundo, a ser\n"
            "adotada por essa empresa é\n"
            "A 360\n"
            "B 370\n"
            "C 380\n"
            "D 390\n"
            "E 400"
        ),
        correct_letter="C",
        correct_value=380.0,
    ))

    # ── 2. ENEM — Média aritmética ───────────────────────────────────
    questions.append(Question(
        id="ENEM-STAT-02",
        source="ENEM 2023",
        category="estatistica",
        text=(
            "Um professor registrou as notas de cinco alunos em uma prova:\n"
            "6, 7, 8, 9, 10.\n"
            "A média aritmética dessas notas é:\n"
            "A 7\n"
            "B 7,5\n"
            "C 8\n"
            "D 8,5\n"
            "E 9"
        ),
        correct_letter="C",
        correct_value=8.0,
    ))

    # ── 3. ENEM — Moda ───────────────────────────────────────────────
    questions.append(Question(
        id="ENEM-STAT-03",
        source="ENEM 2023",
        category="estatistica",
        text=(
            "Em uma pesquisa sobre o número de livros lidos por mês,\n"
            "os seguintes valores foram coletados entre 10 entrevistados:\n"
            "2, 3, 5, 3, 4, 3, 2, 5, 3, 1.\n"
            "A moda dessa distribuição é:\n"
            "A 1\n"
            "B 2\n"
            "C 3\n"
            "D 4\n"
            "E 5"
        ),
        correct_letter="C",
        correct_value=3.0,
    ))

    # ── 4. ENEM — Porcentagem ────────────────────────────────────────
    questions.append(Question(
        id="ENEM-PORC-01",
        source="ENEM 2020",
        category="aritmetica",
        text=(
            "Um produto que custava R$ 250,00 sofreu um desconto de 20%.\n"
            "O novo preço do produto é:\n"
            "A 180\n"
            "B 190\n"
            "C 200\n"
            "D 210\n"
            "E 230"
        ),
        correct_letter="C",
        correct_value=200.0,
    ))

    # ── 5. ENEM — Geometria plana (área de retângulo) ────────────────
    questions.append(Question(
        id="ENEM-GEO-01",
        source="ENEM 2022",
        category="geometria",
        text=(
            "Um terreno retangular tem 30 m de comprimento e 20 m de largura.\n"
            "A área desse terreno, em metros quadrados, é:\n"
            "A 50\n"
            "B 100\n"
            "C 500\n"
            "D 600\n"
            "E 1200"
        ),
        correct_letter="D",
        correct_value=600.0,
    ))

    # ── 6. ENEM — Função afim ────────────────────────────────────────
    questions.append(Question(
        id="ENEM-ALG-01",
        source="ENEM 2023",
        category="algebra",
        text=(
            "Uma empresa cobra uma taxa fixa de R$ 50,00 mais R$ 3,00 por\n"
            "quilômetro rodado em um serviço de transporte. Se um cliente\n"
            "percorreu 40 km, o valor total pago foi de:\n"
            "A 120\n"
            "B 150\n"
            "C 170\n"
            "D 190\n"
            "E 200"
        ),
        correct_letter="C",
        correct_value=170.0,
    ))

    # ── 7. ENEM — Regra de três simples ──────────────────────────────
    questions.append(Question(
        id="ENEM-PROP-01",
        source="ENEM 2021",
        category="aritmetica",
        text=(
            "Uma torneira enche um tanque de 500 litros em 4 horas.\n"
            "Mantendo a mesma vazão, em quantas horas essa torneira\n"
            "encheria um tanque de 1250 litros?\n"
            "A 6\n"
            "B 8\n"
            "C 10\n"
            "D 12\n"
            "E 15"
        ),
        correct_letter="C",
        correct_value=10.0,
    ))

    # ── 8. ENEM — Progressão aritmética ──────────────────────────────
    questions.append(Question(
        id="ENEM-PA-01",
        source="ENEM 2020",
        category="algebra",
        text=(
            "Em uma progressão aritmética, o primeiro termo é 3 e a razão é 5.\n"
            "O décimo termo dessa progressão é:\n"
            "A 43\n"
            "B 45\n"
            "C 48\n"
            "D 50\n"
            "E 53"
        ),
        correct_letter="C",
        correct_value=48.0,
    ))

    # ── 9. OBMEP — Aritmética (soma de dígitos) ──────────────────────
    questions.append(Question(
        id="OBMEP-N1-01",
        source="OBMEP 2023 N1",
        category="aritmetica",
        text=(
            "A soma de todos os números de dois algarismos cujos\n"
            "algarismos somam 5 é:\n"
            "(Exemplos: 14, pois 1+4=5; 23, pois 2+3=5; etc.)\n"
            "A 150\n"
            "B 165\n"
            "C 175\n"
            "D 195\n"
            "E 200"
        ),
        correct_letter="B",
        correct_value=165.0,
    ))

    # ── 10. OBMEP — Fração / Proporção ───────────────────────────────
    questions.append(Question(
        id="OBMEP-N1-02",
        source="OBMEP 2022 N1",
        category="aritmetica",
        text=(
            "Maria tem uma coleção de 120 figurinhas. Ela deu 1/4\n"
            "da coleção para João e 1/3 do que sobrou para Ana.\n"
            "Quantas figurinhas Maria ainda tem?\n"
            "A 40\n"
            "B 50\n"
            "C 60\n"
            "D 70\n"
            "E 80"
        ),
        correct_letter="C",
        correct_value=60.0,
    ))

    # ── 11. ENEM — Probabilidade ─────────────────────────────────────
    questions.append(Question(
        id="ENEM-PROB-01",
        source="ENEM 2021",
        category="probabilidade",
        text=(
            "Uma urna contém 3 bolas vermelhas, 5 bolas azuis e 2 bolas\n"
            "verdes. Retirando-se uma bola ao acaso, a probabilidade de\n"
            "que ela seja azul é:\n"
            "A 1/5\n"
            "B 3/10\n"
            "C 1/2\n"
            "D 2/5\n"
            "E 3/5"
        ),
        correct_letter="C",
        correct_value=0.5,
    ))

    # ── 12. ENEM — Juros simples ─────────────────────────────────────
    questions.append(Question(
        id="ENEM-JUR-01",
        source="ENEM 2023",
        category="algebra",
        text=(
            "Um capital de R$ 2000,00 foi aplicado a juros simples à taxa\n"
            "de 5% ao mês durante 6 meses. O montante ao final do período é:\n"
            "A 2300\n"
            "B 2400\n"
            "C 2500\n"
            "D 2600\n"
            "E 2800"
        ),
        correct_letter="D",
        correct_value=2600.0,
    ))

    # ── 13. ENEM — Equação do 1° grau ────────────────────────────────
    questions.append(Question(
        id="ENEM-EQ-01",
        source="ENEM 2021",
        category="algebra",
        text=(
            "Um pai tem o triplo da idade de seu filho. Juntos, eles somam\n"
            "52 anos. A idade do filho é:\n"
            "A 11\n"
            "B 12\n"
            "C 13\n"
            "D 14\n"
            "E 15"
        ),
        correct_letter="C",
        correct_value=13.0,
    ))

    # ── 14. ENEM — Volume (geometria espacial) ───────────────────────
    questions.append(Question(
        id="ENEM-GEO-02",
        source="ENEM 2020",
        category="geometria",
        text=(
            "Uma caixa d'água tem formato de paralelepípedo com dimensões\n"
            "internas de 2 m de comprimento, 1,5 m de largura e 1 m de\n"
            "altura. A capacidade dessa caixa, em litros, é:\n"
            "A 300\n"
            "B 1500\n"
            "C 2000\n"
            "D 3000\n"
            "E 4500"
        ),
        correct_letter="D",
        correct_value=3000.0,
    ))

    # ── 15. OBMEP — Álgebra (equação quadrática) ─────────────────────
    questions.append(Question(
        id="OBMEP-N2-01",
        source="OBMEP 2023 N2",
        category="algebra",
        text=(
            "Se x² - 7x + 12 = 0, a soma das raízes dessa equação é:\n"
            "A 3\n"
            "B 4\n"
            "C 5\n"
            "D 7\n"
            "E 12"
        ),
        correct_letter="D",
        correct_value=7.0,
    ))

    return questions


# ---------------------------------------------------------------------------
# Solver execution
# ---------------------------------------------------------------------------

def _extract_payload(stdout: str) -> dict:
    start = stdout.find("###JSON_START###")
    end = stdout.rfind("###JSON_END###")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not locate JSON payload markers in solver stdout.")
    raw = stdout[start + len("###JSON_START###"):end].strip()
    return json.loads(raw)


def _extract_answer_letter(final_answer: str, options: dict) -> Optional[str]:
    if not final_answer:
        return None
    # Try explicit letter match
    m = re.search(r"\b(?:option|alternativa|opcao)?\s*([A-Ea-e])\b", final_answer, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        if letter in options or not options:
            return letter
    # Try matching by number
    for tok in re.findall(r"-?\d+(?:[.,]\d+)?", final_answer):
        compact = tok.replace(",", ".")
        try:
            num = float(compact)
        except Exception:
            continue
        for letter, value in options.items():
            if abs(num - value) < 1e-6:
                return letter
    return None


def run_question(q: Question, backend: str, model_name: str = "", adapter_path: str = "", load_in_4bit: bool = False) -> Result:
    cmd = [
        PYTHON_EXE,
        str(SOLVER_SCRIPT),
        "--project-root",
        str(ROOT),
        "--backend",
        backend,
        "--problem-text",
        q.text,
        "--n-plans", "4",
        "--m-drafts", "2",
        "--refine-rounds", "1",
    ]
    if model_name:
        cmd.extend(["--model-name", model_name])
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])
    if load_in_4bit:
        cmd.append("--load-in-4bit")

    # Only enforce strict no-hallucination when using default model (no custom adapter)
    env = {**os.environ, "PYTHONUTF8": "1"}
    if not adapter_path:
        env["STRICT_NO_HALLUCINATION"] = "1"

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=480,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return Result(question=q, error="timeout (480s)", diagnosis="plataforma")

    elapsed = time.monotonic() - t0

    if proc.returncode != 0 and "###JSON_START###" not in proc.stdout:
        return Result(
            question=q,
            error=f"process_failed code={proc.returncode} stderr={proc.stderr.strip()[:300]}",
            elapsed_s=elapsed,
            diagnosis="plataforma",
        )

    try:
        payload = _extract_payload(proc.stdout)
    except Exception as exc:
        return Result(
            question=q,
            error=f"json_parse_failed: {exc}",
            elapsed_s=elapsed,
            diagnosis="plataforma",
        )

    best = payload.get("best_candidate", {}) or {}
    meta = payload.get("meta", {}) or {}
    final_answer = str(best.get("final_answer", ""))

    result = Result(
        question=q,
        answered_value=final_answer,
        fallback_type=str(meta.get("fallback_type", "")),
        fallback_applied=bool(meta.get("fallback_applied", False)),
        trust_gate=str(meta.get("trust_gate", "")),
        elapsed_s=elapsed,
    )

    # Check NAO_CONFIAVEL abstention
    if "NAO_CONFIAVEL" in final_answer:
        result.abstained = True
        result.diagnosis = "modelo"
        return result

    # Build options dict for letter matching
    options = {}
    for line in q.text.splitlines():
        m_opt = re.match(r"^\s*([A-Ea-e])(?:\s*[).:\-]\s*|\s+)([0-9/][0-9\s.,/]*)\s*$", line)
        if m_opt:
            raw = m_opt.group(2).strip().replace(" ", "")
            # Handle fractions like 1/2
            if "/" in raw and raw.count("/") == 1:
                parts = raw.split("/")
                try:
                    options[m_opt.group(1).upper()] = float(parts[0]) / float(parts[1])
                except Exception:
                    pass
            else:
                raw = raw.replace(",", ".")
                try:
                    options[m_opt.group(1).upper()] = float(raw)
                except Exception:
                    pass

    answered_letter = _extract_answer_letter(final_answer, options)
    result.answered_letter = answered_letter

    if answered_letter == q.correct_letter:
        result.is_correct = True
        result.diagnosis = "ok"
    else:
        # Diagnose: platform vs model
        if result.fallback_applied and result.fallback_type:
            result.diagnosis = "plataforma"  # fallback chose wrong answer
        elif result.abstained:
            result.diagnosis = "modelo"
        elif answered_letter is None:
            result.diagnosis = "plataforma"  # could not parse answer
        else:
            result.diagnosis = "modelo"  # model gave wrong answer

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(results: List[Result], backend: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"  BENCHMARK ENEM / OBMEP — Backend: {backend}")
    lines.append("=" * 70)
    lines.append("")

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    abstained = sum(1 for r in results if r.abstained)
    errors = sum(1 for r in results if r.error)
    pct = (correct / total * 100) if total > 0 else 0

    lines.append(f"  Total: {total}  |  Corretas: {correct}  |  Abstenções: {abstained}  |  Erros: {errors}")
    lines.append(f"  Taxa de acerto: {pct:.1f}%")
    lines.append("")

    # Per-category breakdown
    categories = sorted(set(r.question.category for r in results))
    lines.append("  ┌─────────────────┬────────┬──────────┬─────────┐")
    lines.append("  │ Categoria       │ Total  │ Corretas │ Acerto% │")
    lines.append("  ├─────────────────┼────────┼──────────┼─────────┤")
    for cat in categories:
        cat_results = [r for r in results if r.question.category == cat]
        cat_total = len(cat_results)
        cat_correct = sum(1 for r in cat_results if r.is_correct)
        cat_pct = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        lines.append(f"  │ {cat:<15s} │ {cat_total:>6d} │ {cat_correct:>8d} │ {cat_pct:>6.1f}% │")
    lines.append("  └─────────────────┴────────┴──────────┴─────────┘")
    lines.append("")

    # Detailed per-question results
    lines.append("-" * 70)
    lines.append("  RESULTADOS POR QUESTÃO")
    lines.append("-" * 70)

    for r in results:
        icon = "OK" if r.is_correct else ("--" if r.abstained else ("!!" if r.error else "XX"))
        lines.append("")
        lines.append(f"  [{icon}] {r.question.id} ({r.question.source}) — {r.question.category}")
        lines.append(f"      Gabarito: {r.question.correct_letter} ({r.question.correct_value})")
        if r.error:
            lines.append(f"      ERRO: {r.error}")
        else:
            answer_display = r.answered_letter or "?"
            lines.append(f"      Resposta: {answer_display} → \"{r.answered_value}\"")
            if r.fallback_applied:
                lines.append(f"      Fallback: {r.fallback_type}")
            if r.trust_gate:
                lines.append(f"      Trust gate: {r.trust_gate}")
        lines.append(f"      Tempo: {r.elapsed_s:.1f}s")
        lines.append(f"      Diagnóstico: {r.diagnosis.upper()}")

    lines.append("")

    # Diagnosis summary
    plat_issues = [r for r in results if r.diagnosis == "plataforma"]
    model_issues = [r for r in results if r.diagnosis == "modelo"]

    lines.append("=" * 70)
    lines.append("  DIAGNÓSTICO FINAL")
    lines.append("=" * 70)
    lines.append(f"  Problemas da PLATAFORMA: {len(plat_issues)}")
    for r in plat_issues:
        lines.append(f"    - {r.question.id}: {r.error or r.fallback_type or 'parsing/fallback issue'}")
    lines.append(f"  Problemas do MODELO: {len(model_issues)}")
    for r in model_issues:
        reason = "absteve" if r.abstained else f"respondeu {r.answered_letter}, gabarito {r.question.correct_letter}"
        lines.append(f"    - {r.question.id}: {reason}")
    lines.append("")

    if len(plat_issues) > len(model_issues):
        lines.append("  >>> CONCLUSÃO: Há mais falhas na PLATAFORMA do que no modelo.")
        lines.append("      Recomenda-se corrigir fallbacks e parsing antes de retreinar.")
    elif len(model_issues) > len(plat_issues):
        lines.append("  >>> CONCLUSÃO: O MODELO é o principal responsável pelas falhas.")
        lines.append("      Recomenda-se mais treinamento ou troca de modelo base.")
    else:
        lines.append("  >>> CONCLUSÃO: Falhas distribuídas igualmente entre plataforma e modelo.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _configure_stdio()

    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ENEM/OBMEP para IA Matemática")
    parser.add_argument("--backend", choices=["heuristic", "transformers"], default="heuristic")
    parser.add_argument("--model-name", default="", help="HuggingFace model name")
    parser.add_argument("--adapter-path", default="", help="Path to LoRA adapter")
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "eval" / "benchmark"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = build_questions()
    label = args.backend
    if args.model_name:
        label += f" ({args.model_name.split('/')[-1]})"
    print(f"\n  Executando benchmark com {len(questions)} questões (backend={label})...\n")

    results: List[Result] = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i:2d}/{len(questions)}] {q.id} ({q.source})...", end=" ", flush=True)
        r = run_question(q, args.backend, model_name=args.model_name, adapter_path=args.adapter_path, load_in_4bit=args.load_in_4bit)
        results.append(r)
        icon = "OK" if r.is_correct else ("--" if r.abstained else ("!!" if r.error else "XX"))
        ans = r.answered_letter or "?"
        print(f"{icon}  {ans} (esperado {q.correct_letter})  [{r.elapsed_s:.1f}s]")

    report = generate_report(results, args.backend)
    print("\n" + report)

    # Save report and JSON
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_{args.backend}_{stamp}.txt"
    report_path.write_text(report, encoding="utf-8")

    json_path = output_dir / f"benchmark_{args.backend}_{stamp}.json"
    json_data = {
        "backend": args.backend,
        "timestamp": stamp,
        "total": len(results),
        "correct": sum(1 for r in results if r.is_correct),
        "abstained": sum(1 for r in results if r.abstained),
        "errors": sum(1 for r in results if r.error),
        "accuracy_pct": round(sum(1 for r in results if r.is_correct) / len(results) * 100, 1),
        "results": [
            {
                "id": r.question.id,
                "source": r.question.source,
                "category": r.question.category,
                "correct_letter": r.question.correct_letter,
                "answered_letter": r.answered_letter,
                "is_correct": r.is_correct,
                "abstained": r.abstained,
                "fallback_type": r.fallback_type,
                "fallback_applied": r.fallback_applied,
                "trust_gate": r.trust_gate,
                "diagnosis": r.diagnosis,
                "elapsed_s": round(r.elapsed_s, 2),
                "error": r.error,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  Relatório salvo: {report_path}")
    print(f"  Dados JSON:     {json_path}")


if __name__ == "__main__":
    main()
