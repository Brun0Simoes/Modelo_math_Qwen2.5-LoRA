from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional

from .generator import Draft
from .parser import parse_problem
from .router import route_domain
from .schemas import (
    CandidateSolution,
    ParsedProblem,
    SearchSettings,
    SolveRun,
    StrategyPlan,
)
from .strategies import select_strategies
from .verifier import CandidateVerifier


def _extract_final_answer(draft: str) -> str | None:
    patterns = [
        r"final answer\s*:\s*(.+)",
        r"answer\s*:\s*(.+)",
        r"resposta final\s*:\s*(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, draft, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            return ans.splitlines()[0].strip()
    boxed = re.findall(r"\\boxed\{([^{}]{1,220})\}", draft)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in draft.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        if len(last) <= 200 and len(last.split()) <= 36:
            return last
    return None


class CompetitiveSolver:
    def __init__(self, generator, verifier: CandidateVerifier, settings: SearchSettings):
        self.generator = generator
        self.verifier = verifier
        self.settings = settings
        self._current_problem: Optional[ParsedProblem] = None

    def _rank(self, candidates: List[CandidateSolution]) -> List[CandidateSolution]:
        return sorted(candidates, key=lambda c: (c.verifier.passed, c.score), reverse=True)

    def solve(self, problem_text: str) -> SolveRun:
        parsed = parse_problem(problem_text)
        parsed.domain = route_domain(parsed)
        self._current_problem = parsed

        plans = select_strategies(parsed.domain, self.settings.n_plans, parsed.raw_text)
        all_candidates: List[CandidateSolution] = []
        plans_by_name: Dict[str, StrategyPlan] = {p.name: p for p in plans}

        for plan in plans:
            drafts = self.generator.generate(parsed, plan, self.settings.m_drafts)
            all_candidates.extend(self._verify_batch(drafts, plan, round_idx=0))

        for round_idx in range(1, self.settings.refine_rounds + 1):
            ranked = self._rank(all_candidates)
            top = ranked[: self.settings.refine_top_k]
            new_candidates: List[CandidateSolution] = []
            for prev in top:
                plan = plans_by_name[prev.plan_name]
                feedback = "; ".join(prev.verifier.issues[:2]) or "increase rigor and explicit checks"
                refined = self.generator.generate(parsed, plan, n=1, feedback=feedback)
                new_candidates.extend(self._verify_batch(refined, plan, round_idx=round_idx))
            all_candidates.extend(new_candidates)

        ranked = self._rank(all_candidates)
        best = ranked[0]
        return SolveRun(
            problem=parsed,
            settings=self.settings,
            plans=plans,
            candidates=all_candidates,
            best_candidate=best,
        )

    def _verify_batch(
        self,
        drafts: List[Draft],
        plan: StrategyPlan,
        round_idx: int,
    ) -> List[CandidateSolution]:
        if self._current_problem is None:
            raise RuntimeError("Current problem is not set.")
        out: List[CandidateSolution] = []
        for d in drafts:
            report = self.verifier.verify(self._current_problem, d.text)
            out.append(
                CandidateSolution(
                    candidate_id=str(uuid.uuid4()),
                    plan_name=plan.name,
                    round_idx=round_idx,
                    draft=d.text,
                    final_answer=_extract_final_answer(d.text),
                    verifier=report,
                    tool_logs=d.tool_logs,
                )
            )
        return out
