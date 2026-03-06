import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List


ROOT = Path(__file__).resolve().parents[1]
SOLVER_SCRIPT = ROOT / "scripts" / "solve_problem_json.py"
PYTHON_EXE = sys.executable


@dataclass
class Case:
    name: str
    text: str
    expected_fallback: str
    expected_number: int | None = None
    expected_phrase: str | None = None


def _extract_payload(stdout: str) -> dict:
    start = stdout.find("###JSON_START###")
    end = stdout.rfind("###JSON_END###")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not find JSON markers in solver output.")
    raw = stdout[start + len("###JSON_START###") : end].strip()
    return json.loads(raw)


def _numbers(text: str) -> List[float]:
    values = []
    for tok in re.findall(r"-?\d[\d\s.,]*", str(text or "")):
        compact = tok.replace("\u00a0", " ")
        compact = re.sub(r"(?<=\d)[\s.](?=\d{3}\b)", "", compact)
        compact = re.sub(r"[^0-9,.\-]", "", compact)
        if not compact:
            continue
        if compact.count(",") == 1 and compact.count(".") <= 1:
            compact = compact.replace(".", "").replace(",", ".")
        else:
            compact = compact.replace(",", "")
            if compact.count(".") > 1:
                compact = compact.replace(".", "")
        try:
            values.append(float(compact))
        except Exception:
            continue
    return values


def _is_complete_answer(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if s.endswith(("+", "-", "*", "/", "=", "{", "(", "\\", "_", "^", ":")):
        return False
    if s.count("{") != s.count("}"):
        return False
    if s.count("(") != s.count(")"):
        return False
    return True


def run_case(case: Case) -> tuple[bool, str]:
    cmd = [
        PYTHON_EXE,
        str(SOLVER_SCRIPT),
        "--project-root",
        str(ROOT),
        "--backend",
        "transformers",
        "--problem-text",
        case.text,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return False, f"process_failed code={proc.returncode} stderr={proc.stderr.strip()[:220]}"

    try:
        payload = _extract_payload(proc.stdout)
    except Exception as exc:
        return False, f"json_parse_failed {exc}"

    meta = payload.get("meta", {}) or {}
    best = payload.get("best_candidate", {}) or {}
    final_answer = str(best.get("final_answer", ""))

    if str(meta.get("fallback_type")) != case.expected_fallback:
        return False, f"fallback_type mismatch got={meta.get('fallback_type')}"
    if not bool(meta.get("fallback_applied", False)):
        return False, "fallback_applied is False"
    if not _is_complete_answer(final_answer):
        return False, f"incomplete final_answer={final_answer!r}"

    if case.expected_number is not None:
        nums = _numbers(final_answer)
        if not any(abs(n - case.expected_number) < 1e-6 for n in nums):
            return False, f"expected number {case.expected_number} not found in final_answer={final_answer!r}"

    if case.expected_phrase is not None:
        if case.expected_phrase.lower() not in final_answer.lower():
            return False, f"expected phrase missing: {case.expected_phrase!r}"

    return True, "ok"


def mutate_text(base: str, transforms: List[Callable[[str], str]]) -> List[str]:
    texts = [base]
    for transform in transforms:
        texts.append(transform(base))
    dedup = []
    seen = set()
    for t in texts:
        key = t.strip()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)
    return dedup


def build_cases() -> List[Case]:
    cases: List[Case] = []

    fuel_a = """Uma distribuidora de combustivel possui caminhoes-tanque com capacidade de 30 000 litros cada.
Em qualquer transporte, um mesmo volume de combustivel e descartado por impurezas.
Um posto encomendou 10 000 litros, a distribuidora enviou 10 200 litros.
A quantidade entregue foi 9 900 litros.
Em novo pedido, o posto solicitou exatamente o dobro do volume encomendado no pedido anterior.
Qual o volume minimo a enviar para garantir a entrega?
A 20100
B 20200
C 20300
D 20400
E 20600"""
    fuel_b = """Uma distribuidora sempre perde o mesmo volume por transporte.
Um posto encomendou 8 000 litros, foram enviados 8 350 litros e entregues 7 950 litros.
No novo pedido, deseja-se entregar exatamente o dobro do volume encomendado anteriormente.
Qual o volume minimo que deve ser enviado?
A 16300
B 16400
C 16500
D 16600
E 16700"""

    fuel_transforms = [
        lambda t: t.replace("Qual o volume minimo a enviar para garantir a entrega?", "Qual e o volume minimo a enviar para garantir a entrega?"),
        lambda t: t.replace("dobro", "DOBRO"),
        lambda t: t.replace("encomendou", "encomendou\n"),
        lambda t: t.replace("litros", " litros "),
        lambda t: t.replace("distribuidora", "Distribuidora"),
        lambda t: t.replace("caminhoes-tanque", "caminhoes tanque"),
        lambda t: t.replace("impurezas", "impurezas."),
        lambda t: t.replace("A 20100", "A) 20100"),
        lambda t: t.replace("B 16400", "B. 16400"),
    ]

    for i, text in enumerate(mutate_text(fuel_a, fuel_transforms), start=1):
        cases.append(Case(f"fuel-a-{i:02d}", text, "constant_loss_transport", expected_number=20300))
    for i, text in enumerate(mutate_text(fuel_b, fuel_transforms), start=1):
        cases.append(Case(f"fuel-b-{i:02d}", text, "constant_loss_transport", expected_number=16400))

    cyc = """Problema (Desigualdades / Algebra Simetrica)
Sejam a, b, c > 0. Prove que
a/sqrt(a^2+8bc) + b/sqrt(b^2+8ca) + c/sqrt(c^2+8ab) >= 1."""
    cyc_transforms = [
        lambda t: t.replace("Prove", "Frove"),
        lambda t: t.replace("sqrt", "√"),
        lambda t: t.replace("^2", "²"),
        lambda t: t.replace(">=", "≥"),
        lambda t: t.replace("a, b, c", "a,b,c"),
        lambda t: t.replace("Sejam", "Demonstre para"),
        lambda t: t.replace("Problema (Desigualdades / Algebra Simetrica)\n", ""),
        lambda t: t.replace(" + ", "+"),
        lambda t: t.replace("b/sqrt(b^2+8ca)", "b / sqrt(b^2+8ca)"),
    ]
    for i, text in enumerate(mutate_text(cyc, cyc_transforms), start=1):
        cases.append(
            Case(
                f"cyclic-{i:02d}",
                text,
                "cyclic_sqrt8bc_ge_1",
                expected_phrase="minimum value 1",
            )
        )

    circle_a = """QUESTAO 136
No entorno de uma lagoa circular, de raio 1 km, ha uma ciclovia.
Para todo ponto da ciclovia, e considerado protegido se houver pelo menos um policial, no maximo, 200 m daquele ponto, posicionado sobre a ciclovia.
Utilize 3 como aproximacao para pi.
Nessas condicoes, a quantidade minima necessaria de policiais e:
A 4
B 8
C 15
D 30
E 60"""

    circle_b = """Uma pista circular de raio 2 km deve ser protegida por policiais.
Um ponto da pista esta protegido se houver policial a no maximo 250 m desse ponto sobre a pista.
Use pi = 3.
Qual a quantidade minima de policiais?
A 12
B 20
C 24
D 28
E 32"""

    circle_transforms = [
        lambda t: t.replace("maximo", "mAximo"),
        lambda t: t.replace("policial", "policiaI"),
        lambda t: t.replace("ciclovia", "ciclo via"),
        lambda t: t.replace("aproximacao para pi", "aproximacao para o pi"),
        lambda t: t.replace("A 4", "A) 4"),
        lambda t: t.replace("B 8", "B. 8"),
        lambda t: t.replace("C 15", "C: 15"),
        lambda t: t.replace("raio 1 km", "raio de 1 km"),
        lambda t: t.replace("200 m", "200m"),
        lambda t: t.replace("Use pi = 3.", "Use pi ~= 3."),
    ]

    for i, text in enumerate(mutate_text(circle_a, circle_transforms), start=1):
        cases.append(Case(f"circle-a-{i:02d}", text, "circular_ciclovia_coverage", expected_number=15))
    for i, text in enumerate(mutate_text(circle_b, circle_transforms), start=1):
        cases.append(Case(f"circle-b-{i:02d}", text, "circular_ciclovia_coverage", expected_number=24))

    return cases


def main() -> None:
    cases = build_cases()
    failures = []
    passed = 0

    for case in cases:
        ok, reason = run_case(case)
        if ok:
            passed += 1
            print(f"[PASS] {case.name}")
        else:
            failures.append((case.name, reason))
            print(f"[FAIL] {case.name}: {reason}")

    total = len(cases)
    print("\n=== Regression Summary ===")
    print(f"total={total} passed={passed} failed={len(failures)}")

    if failures:
        print("\nFailed cases:")
        for name, reason in failures[:20]:
            print(f"- {name}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
