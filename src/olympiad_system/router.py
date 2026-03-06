from __future__ import annotations

from typing import Dict

from .schemas import ParsedProblem


DOMAIN_KEYWORDS: Dict[str, set[str]] = {
    "algebra": {
        "equation",
        "inequality",
        "polynomial",
        "function",
        "roots",
        "factor",
        "matrix",
        "alg",
    },
    "number_theory": {
        "integer",
        "prime",
        "divisible",
        "divisibility",
        "mod",
        "modulo",
        "gcd",
        "lcm",
        "congruence",
        "nt",
    },
    "combinatorics": {
        "combinatorics",
        "count",
        "arrangement",
        "permutation",
        "combination",
        "graph",
        "pigeonhole",
        "invariant",
    },
    "geometry": {
        "triangle",
        "circle",
        "angle",
        "parallel",
        "perpendicular",
        "area",
        "length",
        "distance",
        "geometry",
    },
}


def route_domain(problem: ParsedProblem) -> str:
    text = problem.raw_text.lower()
    scores: Dict[str, int] = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, words in DOMAIN_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[domain] += 1
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "mixed"
    return best_domain
