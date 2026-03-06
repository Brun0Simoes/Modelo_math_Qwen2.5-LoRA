from __future__ import annotations

import re
from typing import List, Tuple

from .schemas import ParsedProblem


OBJECTIVE_PATTERNS = [
    r"\bprove\b",
    r"\bshow\b",
    r"\bfind\b",
    r"\bdetermine\b",
    r"\bcompute\b",
    r"\bevaluate\b",
    r"\bsolve\b",
    r"\bmostre\b",
    r"\bprove que\b",
    r"\bencontre\b",
    r"\bdetermine\b",
    r"\bcalcule\b",
    r"\bresolva\b",
    r"\bqual\b",
    r"\bquanto\b",
    r"\bminimum\b",
    r"\bminimo\b",
]


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[\.\?\!;])\s+|\n+", text)
    return [c.strip() for c in chunks if c.strip()]


def _is_option_line(line: str) -> bool:
    # Multiple-choice line like "A 20100"
    return bool(re.match(r"^\s*[A-Ea-e]\s+\d", line))


def _locate_objective(sentences: List[str]) -> Tuple[str, int]:
    lowered = [s.lower() for s in sentences]
    for pattern in OBJECTIVE_PATTERNS:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        for idx, s in enumerate(lowered):
            if _is_option_line(sentences[idx]):
                continue
            if rx.search(s):
                return sentences[idx], idx

    for idx in range(len(sentences) - 1, -1, -1):
        if not _is_option_line(sentences[idx]):
            return sentences[idx], idx
    return "", -1


def _extract_variables(text: str) -> List[str]:
    vars_found = set()
    for token in re.findall(r"\b[a-zA-Z]\w*\b", text):
        if token.lower() in {"and", "or", "if", "then", "for", "with", "let"}:
            continue
        if len(token) <= 3:
            vars_found.add(token)
    return sorted(vars_found)


def parse_problem(problem_text: str) -> ParsedProblem:
    sentences = _split_sentences(problem_text)
    objective, obj_idx = _locate_objective(sentences)
    hypotheses = []
    if obj_idx > 0:
        hypotheses = sentences[:obj_idx]
    elif len(sentences) > 1:
        hypotheses = sentences[:-1]
    elif sentences:
        hypotheses = [sentences[0]]
    variables = _extract_variables(problem_text)
    return ParsedProblem(
        raw_text=problem_text.strip(),
        objective=objective.strip(),
        hypotheses=hypotheses,
        variables=variables,
    )
