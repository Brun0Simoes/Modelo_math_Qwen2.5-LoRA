from __future__ import annotations

import hashlib
import random
from typing import Dict, List

from .schemas import StrategyPlan


STRATEGY_BANK: Dict[str, List[StrategyPlan]] = {
    "algebra": [
        StrategyPlan(
            name="Substitution",
            rationale="Reduce degrees of freedom by replacing variables.",
            prompt_template="Try substitutions that simplify symmetric/asymmetric terms.",
        ),
        StrategyPlan(
            name="Inequality Toolkit",
            rationale="Apply AM-GM, Cauchy-Schwarz, Jensen when structure matches.",
            prompt_template="Detect convexity or product-sum patterns and test standard inequalities.",
        ),
        StrategyPlan(
            name="Polynomial Structure",
            rationale="Use factorization and root constraints.",
            prompt_template="Rearrange into polynomial identity and inspect roots/multiplicities.",
        ),
        StrategyPlan(
            name="Functional Equation Cases",
            rationale="Probe special values and injectivity/surjectivity.",
            prompt_template="Set strategic values (0,1,-1), then test composition patterns.",
        ),
    ],
    "number_theory": [
        StrategyPlan(
            name="Modular Sweep",
            rationale="Scan small moduli to kill impossible cases quickly.",
            prompt_template="Try mod 2,3,4,5,8,9,11 and keep only surviving residue classes.",
        ),
        StrategyPlan(
            name="Valuation and Divisibility",
            rationale="Use p-adic valuations and divisibility chains.",
            prompt_template="Track prime exponents and compare both sides term by term.",
        ),
        StrategyPlan(
            name="Bounding + Contradiction",
            rationale="Combine inequalities with divisibility constraints.",
            prompt_template="Assume minimal counterexample and derive contradiction via bounds.",
        ),
        StrategyPlan(
            name="Chinese Remainder Decomposition",
            rationale="Split into independent congruence systems.",
            prompt_template="Solve on coprime components and stitch back with CRT.",
        ),
    ],
    "combinatorics": [
        StrategyPlan(
            name="Invariant/Half-Invariant",
            rationale="Track quantity preserved under operations.",
            prompt_template="Propose an invariant and test move-by-move behavior.",
        ),
        StrategyPlan(
            name="Extremal Principle",
            rationale="Choose maximal/minimal object and force structure.",
            prompt_template="Assume extremal object and derive deterministic consequences.",
        ),
        StrategyPlan(
            name="Double Counting",
            rationale="Count the same set in two ways.",
            prompt_template="Define incidence structure and compare two counts.",
        ),
        StrategyPlan(
            name="Probabilistic Method",
            rationale="Show existence by expectation.",
            prompt_template="Construct random object and prove expected value threshold.",
        ),
    ],
    "geometry": [
        StrategyPlan(
            name="Coordinate Bash",
            rationale="Convert geometry constraints to algebraic equations.",
            prompt_template="Choose coordinates to simplify lines/circles and compute explicitly.",
        ),
        StrategyPlan(
            name="Angle Chasing",
            rationale="Exploit cyclic/quadrilateral angle relations.",
            prompt_template="List known equal angles and propagate via cyclic lemmas.",
        ),
        StrategyPlan(
            name="Transformations",
            rationale="Use inversion/homothety/reflection when symmetry appears.",
            prompt_template="Apply a transformation that linearizes the hard relation.",
        ),
        StrategyPlan(
            name="Vector/Barycentric",
            rationale="Rewrite geometric statements in vector form.",
            prompt_template="Set vectors at a convenient origin and reduce to dot/cross products.",
        ),
    ],
    "mixed": [
        StrategyPlan(
            name="Case Split",
            rationale="Separate by parity/sign/ordering to simplify logic.",
            prompt_template="Split into minimal complete cases and solve each cleanly.",
        ),
        StrategyPlan(
            name="Constructive + Verify",
            rationale="Build candidate and verify rigorously.",
            prompt_template="Guess a structure from small examples, then prove by induction or contradiction.",
        ),
        StrategyPlan(
            name="Search for Counterexample",
            rationale="Try to falsify quickly before proving.",
            prompt_template="Run small-value stress tests; if none fail, formalize the discovered pattern.",
        ),
        StrategyPlan(
            name="Backward from Goal",
            rationale="Work backward from target identity/inequality.",
            prompt_template="Transform target into equivalent statement tied to givens.",
        ),
    ],
}


def select_strategies(domain: str, n_plans: int, seed_text: str) -> List[StrategyPlan]:
    pool = STRATEGY_BANK.get(domain, STRATEGY_BANK["mixed"]).copy()
    if not pool:
        pool = STRATEGY_BANK["mixed"].copy()
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    rng.shuffle(pool)
    if n_plans <= len(pool):
        return pool[:n_plans]
    out = []
    while len(out) < n_plans:
        for p in pool:
            out.append(p)
            if len(out) >= n_plans:
                break
    return out
