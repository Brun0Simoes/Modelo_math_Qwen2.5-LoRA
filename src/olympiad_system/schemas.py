from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedProblem:
    raw_text: str
    objective: str
    hypotheses: List[str]
    variables: List[str]
    domain: str = "mixed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyPlan:
    name: str
    rationale: str
    prompt_template: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    score: float
    passed: bool
    issues: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateSolution:
    candidate_id: str
    plan_name: str
    round_idx: int
    draft: str
    final_answer: Optional[str]
    verifier: VerificationReport
    tool_logs: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.verifier.score

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = self.score
        return data


@dataclass
class SearchSettings:
    n_plans: int = 8
    m_drafts: int = 4
    refine_rounds: int = 1
    refine_top_k: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolveRun:
    problem: ParsedProblem
    settings: SearchSettings
    plans: List[StrategyPlan]
    candidates: List[CandidateSolution]
    best_candidate: CandidateSolution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "settings": self.settings.to_dict(),
            "plans": [p.to_dict() for p in self.plans],
            "candidates": [c.to_dict() for c in self.candidates],
            "best_candidate": self.best_candidate.to_dict(),
        }
