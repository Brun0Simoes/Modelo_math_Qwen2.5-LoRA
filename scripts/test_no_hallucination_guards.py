from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"E:\IA_matematica")
SOLVER_MODULE_PATH = PROJECT_ROOT / "scripts" / "solve_problem_json.py"
SOLVER_SCRIPT = PROJECT_ROOT / "scripts" / "solve_problem_json.py"


def _load_solver_module():
    spec = importlib.util.spec_from_file_location("solve_problem_json", SOLVER_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SOLVER_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _extract_payload(stdout: str) -> dict:
    start = stdout.find("###JSON_START###")
    end = stdout.rfind("###JSON_END###")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not locate JSON payload markers.")
    raw = stdout[start + len("###JSON_START###") : end].strip()
    return json.loads(raw)


def test_option_matching_and_consensus(mod) -> None:
    options = {"A": 360.0, "B": 370.0, "C": 380.0, "D": 390.0, "E": 400.0}

    _assert(
        not mod._final_matches_options("a/b + c", options),
        "Variable-only expression should not match MCQ options.",
    )
    _assert(
        mod._final_matches_options("C (380)", options),
        "Canonical option answer should match options.",
    )

    payload = {
        "candidates": [
            {
                "final_answer": "C (380)",
                "draft": "Final answer: C (380)",
                "verifier": {"score": 0.85, "passed": True},
                "score": 0.85,
            },
            {
                "final_answer": "380",
                "draft": "Final answer: 380",
                "verifier": {"score": 0.78, "passed": True},
                "score": 0.78,
            },
            {
                "final_answer": "B (370)",
                "draft": "Final answer: B (370)",
                "verifier": {"score": 0.20, "passed": False},
                "score": 0.20,
            },
        ]
    }
    consensus = mod._infer_option_consensus(payload, options)
    _assert(consensus is not None, "Consensus should exist.")
    _assert(consensus["letter"] == "C", "Consensus should select option C.")


def test_trust_gate_primitives(mod) -> None:
    options = {"A": 20100.0, "B": 20200.0, "C": 20300.0, "D": 20400.0, "E": 20600.0}
    good = {
        "final_answer": "20300 (option C)",
        "draft": "Strategy: sanity\nFinal answer: 20300 (option C)",
        "verifier": {"score": 0.93, "passed": True},
        "score": 0.93,
    }
    bad = {
        "final_answer": "\\frac{a}{",
        "draft": "Strategy: test\nFinal answer: \\frac{a}{",
        "verifier": {"score": 0.96, "passed": True},
        "score": 0.96,
    }
    _assert(mod._is_trustworthy_candidate(good, options), "Good objective candidate should be trustworthy.")
    _assert(not mod._is_trustworthy_candidate(bad, options), "Truncated candidate must be untrustworthy.")

    abstain = mod._make_abstain_candidate("unit_test")
    _assert("NAO_CONFIAVEL" in str(abstain.get("final_answer", "")), "Abstain candidate marker missing.")


def test_verifier_mcq_behavior() -> None:
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from olympiad_system.parser import parse_problem
    from olympiad_system.tools import ToolSandbox
    from olympiad_system.verifier import CandidateVerifier

    text = """Uma empresa quer adotar uma velocidade de referencia.
Valores observados: 390, 380, 320, 390, 340, 380, 390, 400, 350, 360.
Qual valor deve ser adotado?
A 360
B 370
C 380
D 390
E 400"""
    problem = parse_problem(text)
    verifier = CandidateVerifier(ToolSandbox())

    wrong_report = verifier.verify(problem, "Strategy: quick\nFinal answer: 999")
    _assert(not wrong_report.passed, "Invalid MCQ answer should fail verifier.")

    ok_draft = "\n".join(
        [
            "Strategy: Ordered statistics",
            "Step 1: Sort values and identify the central result.",
            "Step 2: Validate with the listed options.",
            "Final answer: C (380)",
        ]
    )
    ok_report = verifier.verify(problem, ok_draft)
    _assert(ok_report.passed, "Valid MCQ option should pass verifier.")


def test_main_strict_abstain_heuristic() -> None:
    problem = """Uma loja registrou cinco medidas de venda em dias distintos.
Dados: 11, 15, 19, 14, 20.
Qual alternativa representa a soma dessas medidas?
A 70
B 75
C 79
D 81
E 90"""
    cmd = [
        sys.executable,
        str(SOLVER_SCRIPT),
        "--project-root",
        str(PROJECT_ROOT),
        "--backend",
        "heuristic",
        "--problem-text",
        problem,
        "--n-plans",
        "2",
        "--m-drafts",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "STRICT_NO_HALLUCINATION": "1"},
    )
    _assert(proc.returncode == 0, f"Solver script failed in heuristic mode: {proc.stderr[:300]}")
    payload = _extract_payload(proc.stdout)
    best = payload.get("best_candidate") or {}
    meta = payload.get("meta") or {}
    _assert("NAO_CONFIAVEL" in str(best.get("final_answer", "")), "Strict trust gate should abstain when untrusted.")
    _assert(meta.get("fallback_type") == "safety_abstain", "Expected strict safety abstain fallback.")


def main() -> None:
    mod = _load_solver_module()
    test_option_matching_and_consensus(mod)
    test_trust_gate_primitives(mod)
    test_verifier_mcq_behavior()
    test_main_strict_abstain_heuristic()
    print("All no-hallucination guard tests passed.")


if __name__ == "__main__":
    main()
