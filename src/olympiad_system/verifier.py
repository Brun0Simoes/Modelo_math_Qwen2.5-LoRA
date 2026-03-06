from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import sympy as sp

from .schemas import ParsedProblem, VerificationReport
from .tools import ToolSandbox, parse_math_expr


FINAL_ANSWER_PATTERNS = [
    r"final answer\s*:\s*(.+)",
    r"answer\s*:\s*(.+)",
    r"resposta final\s*:\s*(.+)",
]

RED_FLAGS = [
    "obviously",
    "clearly true",
    "sem perda de generalidade",
    "without loss of generality",
]

REASONING_MARKERS = [
    "therefore",
    "thus",
    "hence",
    "logo",
    "we claim",
    "it follows",
    "suppose",
    "assume",
]

PROOF_MARKERS = [
    "contradiction",
    "induction",
    "invariant",
    "extremal",
    "double counting",
    "bijection",
    "wlog",
    "without loss of generality",
]

PROOF_OBJECTIVE_MARKERS = [
    "prove",
    "show",
    "demonstrate",
    "mostre",
    "prove que",
]

VALUE_OBJECTIVE_MARKERS = [
    "find",
    "determine",
    "compute",
    "evaluate",
    "solve",
    "encontre",
    "determine",
    "calcule",
    "resolva",
]


def _parse_number_token(token: str) -> Optional[float]:
    raw = str(token or "").strip().replace(" ", "")
    raw = raw.replace(",", ".")
    try:
        return float(raw)
    except Exception:
        return None


def _extract_options(problem_text: str) -> Dict[str, float]:
    options: Dict[str, float] = {}
    for line in str(problem_text or "").splitlines():
        m = re.match(
            r"^\s*([A-Ea-e])(?:\s*[).:\-]\s*|\s+)([0-9][0-9\s.,]*)\s*$",
            line,
        )
        if not m:
            continue
        value = _parse_number_token(m.group(2))
        if value is None:
            continue
        options[m.group(1).upper()] = value
    return options


def _extract_option_vote(text: str, options: Dict[str, float]) -> Optional[str]:
    if not options:
        return None
    content = str(text or "")
    m = re.search(r"\b(?:option|alternativa|opcao)\s*([A-Ea-e])\b", content, flags=re.IGNORECASE)
    if m is None:
        m = re.match(r"^\s*([A-Ea-e])\s*(?:[\).:\-]|\(|$)", content.strip())
    if m:
        letter = m.group(1).upper()
        if letter in options:
            return letter
    for tok in re.findall(r"-?\d+(?:[.,]\d+)?", content):
        val = _parse_number_token(tok)
        if val is None:
            continue
        for letter, opt_val in options.items():
            if abs(val - opt_val) <= 1e-6:
                return letter
    return None


def _looks_truncated_answer(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if s.count("{") != s.count("}"):
        return True
    if s.count("(") != s.count(")"):
        return True
    if s.endswith(("+", "-", "*", "/", "=", "{", "(", "\\", "_", "^", ":", ",")):
        return True
    return False


def _extract_final_answer(text: str) -> Optional[str]:
    lowered = text.lower()
    for pattern in FINAL_ANSWER_PATTERNS:
        rx = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        m = rx.search(lowered)
        if m:
            ans = text[m.start(1) : m.end(1)].strip()
            return ans.splitlines()[0].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    if len(last) <= 120:
        return last
    return None


def _extract_assignments(text: str) -> Dict[str, float]:
    pairs = re.findall(r"\b([a-zA-Z]\w*)\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\b", text)
    out: Dict[str, float] = {}
    for var, value in pairs:
        try:
            out[var] = float(sp.N(parse_math_expr(value)))
        except Exception:
            continue
    return out


def _extract_numeric_values(text: str) -> List[float]:
    values: List[float] = []
    for token in re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text):
        try:
            values.append(float(sp.N(parse_math_expr(token))))
        except Exception:
            continue
    return values


def _objective_has_any(text: str, markers: List[str]) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in markers)


def _infer_equation_variables(equations: List[str]) -> List[str]:
    vars_found = set()
    for eq in equations:
        try:
            lhs, _, rhs = _relation_to_expr(eq)
            for sym in lhs.free_symbols.union(rhs.free_symbols):
                vars_found.add(str(sym))
        except Exception:
            continue
    return sorted(vars_found)


def _infer_assignments_from_answer(final_answer: str, variables: List[str]) -> Dict[str, float]:
    if not final_answer or not variables:
        return {}
    # Tuple-style answer: (x, y, z)
    tuple_match = re.search(r"\(([^()]*)\)", final_answer)
    if tuple_match:
        raw = [p.strip() for p in tuple_match.group(1).split(",") if p.strip()]
        if len(raw) == len(variables):
            out: Dict[str, float] = {}
            for var, token in zip(variables, raw):
                try:
                    out[var] = float(sp.N(parse_math_expr(token)))
                except Exception:
                    return {}
            return out
    # Single-value answer maps to one-variable equations.
    if len(variables) == 1:
        nums = _extract_numeric_values(final_answer)
        if nums:
            return {variables[0]: nums[-1]}
    return {}


def _relation_to_expr(equation: str) -> Tuple[sp.Expr, str, sp.Expr]:
    for op in ("==", ">=", "<=", "=", ">", "<"):
        if op in equation:
            left, right = equation.split(op, 1)
            return parse_math_expr(left.strip()), op, parse_math_expr(right.strip())
    raise ValueError(f"Invalid relation: {equation}")


class CandidateVerifier:
    def __init__(self, tool_sandbox: ToolSandbox) -> None:
        self.tools = tool_sandbox

    def verify(self, problem: ParsedProblem, draft: str) -> VerificationReport:
        score = 0.1
        issues: List[str] = []
        checks: Dict[str, object] = {}
        hard_fail = False

        final_answer = _extract_final_answer(draft)
        checks["final_answer"] = final_answer
        if final_answer:
            score += 0.1
        else:
            issues.append("Missing explicit final answer.")

        draft_lower = draft.lower()
        penalties = 0.0
        for flag in RED_FLAGS:
            if flag in draft_lower:
                penalties += 0.08
                issues.append(f"Weak justification phrase found: '{flag}'.")

        non_empty_lines = [line.strip() for line in draft.splitlines() if line.strip()]
        if len(non_empty_lines) >= 4:
            score += 0.05
        if "strategy:" in draft_lower:
            score += 0.05
        reasoning_hits = sum(1 for marker in REASONING_MARKERS if marker in draft_lower)
        score += min(0.12, 0.03 * reasoning_hits)
        if any(marker in draft_lower for marker in PROOF_MARKERS):
            score += 0.05
        if "=" in draft or "\\boxed{" in draft or "=>" in draft or "\\implies" in draft:
            score += 0.05

        equations = self.tools.extract_equations(problem.raw_text)
        checks["problem_equations"] = equations
        assignments = _extract_assignments(final_answer or draft)
        objective_text = problem.objective or problem.raw_text
        proof_like = _objective_has_any(objective_text, PROOF_OBJECTIVE_MARKERS)
        value_like = _objective_has_any(objective_text, VALUE_OBJECTIVE_MARKERS)
        options = _extract_options(problem.raw_text)
        objective_mcq = bool(options) and not proof_like
        eq_variables = _infer_equation_variables(equations)
        if not assignments and final_answer:
            inferred = _infer_assignments_from_answer(final_answer, eq_variables)
            if inferred:
                assignments = inferred
        checks["assignments"] = assignments
        checks["objective_mcq"] = objective_mcq
        checks["options"] = options

        if equations and assignments:
            eq_results = []
            all_good = True
            for eq in equations:
                try:
                    lhs, op, rhs = _relation_to_expr(eq)
                    subs = {sp.Symbol(k): v for k, v in assignments.items()}
                    lhs_v = float(sp.N(lhs.subs(subs)))
                    rhs_v = float(sp.N(rhs.subs(subs)))
                    if op in ("=", "=="):
                        ok = abs(lhs_v - rhs_v) < 1e-8
                    elif op == ">=":
                        ok = lhs_v >= rhs_v - 1e-8
                    elif op == "<=":
                        ok = lhs_v <= rhs_v + 1e-8
                    elif op == ">":
                        ok = lhs_v > rhs_v
                    else:
                        ok = lhs_v < rhs_v
                    eq_results.append({"eq": eq, "ok": ok, "lhs": lhs_v, "rhs": rhs_v})
                    if not ok:
                        all_good = False
                except Exception as exc:
                    eq_results.append({"eq": eq, "ok": False, "error": str(exc)})
                    all_good = False
            checks["equation_checks"] = eq_results
            if all_good:
                score += 0.45
            else:
                penalties += 0.25
                issues.append("Final assignment fails one or more original constraints.")
                hard_fail = True
        elif equations and not assignments:
            if proof_like and not value_like:
                # For theorem/inequality proofs, equation extraction should not force
                # a numeric assignment-style final answer.
                if final_answer and len(non_empty_lines) >= 3:
                    score += 0.1
                if any(tok in draft_lower for tok in ["case", "suppose", "let ", "consider"]):
                    score += 0.05
                if any(tok in draft_lower for tok in ["verify", "check", "constraint", "consisten"]):
                    score += 0.05
            else:
                issues.append("Could not parse numeric assignment for equation checking.")
        elif not equations:
            # For proof-style prompts, score reasoning quality instead of assignment checks.
            if final_answer and len(non_empty_lines) >= 3:
                score += 0.1
            if any(tok in draft_lower for tok in ["case", "suppose", "let ", "consider"]):
                score += 0.05
            if any(tok in draft_lower for tok in ["verify", "check", "constraint", "consisten"]):
                score += 0.05

        if final_answer and _looks_truncated_answer(final_answer):
            penalties += 0.3
            issues.append("Final answer appears truncated or syntactically incomplete.")
            hard_fail = True

        if objective_mcq:
            vote = _extract_option_vote(final_answer or draft, options)
            checks["option_vote"] = vote
            if vote is None:
                penalties += 0.55
                issues.append("Objective multiple-choice answer does not match any option.")
                hard_fail = True
            else:
                score += 0.18
            if "unable to determine" in (final_answer or "").lower():
                penalties += 0.6
                issues.append("Abstained answer for objective multiple-choice question.")
                hard_fail = True

        score = max(0.0, min(1.0, score - penalties))
        needs_numeric_solution = bool(equations) and not (proof_like and not value_like)
        pass_threshold = 0.6 if needs_numeric_solution else 0.45
        checks["pass_threshold"] = pass_threshold
        passed = score >= pass_threshold and not hard_fail
        return VerificationReport(score=score, passed=passed, issues=issues, checks=checks)
