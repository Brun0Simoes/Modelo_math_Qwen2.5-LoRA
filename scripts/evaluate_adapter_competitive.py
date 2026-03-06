import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def ensure_inside_root(path: Path, project_root: Path) -> None:
    if not str(path.resolve()).lower().startswith(str(project_root.resolve()).lower()):
        raise ValueError(f"Path must be inside project root: {path}")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(
    project_root: Path,
    model_name: str,
    adapter_path: str,
    problems: List[str],
    max_new_tokens: int,
    temperature: float,
    n_plans: int,
    m_drafts: int,
    refine_rounds: int,
    refine_top_k: int,
) -> dict:
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from olympiad_system import (
        CandidateVerifier,
        CompetitiveSolver,
        SearchSettings,
        ToolSandbox,
        TransformersGenerator,
    )

    tools = ToolSandbox()
    verifier = CandidateVerifier(tools)
    generator = TransformersGenerator(
        model_name=model_name,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    settings = SearchSettings(
        n_plans=n_plans,
        m_drafts=m_drafts,
        refine_rounds=refine_rounds,
        refine_top_k=refine_top_k,
    )
    solver = CompetitiveSolver(generator=generator, verifier=verifier, settings=settings)

    runs = []
    passed = 0
    score_sum = 0.0
    for i, problem in enumerate(problems):
        run = solver.solve(problem)
        best = run.best_candidate
        score_sum += best.score
        if best.verifier.passed:
            passed += 1
        runs.append(
            {
                "index": i,
                "score": best.score,
                "passed": best.verifier.passed,
                "plan": best.plan_name,
                "final_answer": best.final_answer,
            }
        )
    n = len(problems)
    return {
        "adapter_path": adapter_path,
        "n_problems": n,
        "pass_rate": (passed / n) if n else 0.0,
        "avg_score": (score_sum / n) if n else 0.0,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument("--input-jsonl", default="E:\\IA_matematica\\data\\processed\\curriculum\\stage3_competition.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--adapters", required=True, help="Comma-separated adapter directories.")
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-plans", type=int, default=4)
    parser.add_argument("--m-drafts", type=int, default=2)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--refine-top-k", type=int, default=2)
    parser.add_argument("--output-dir", default="E:\\IA_matematica\\outputs\\eval\\adapter_compare")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    ensure_inside_root(input_path, project_root)
    ensure_inside_root(output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    random.Random(args.seed).shuffle(rows)
    problems = [(r.get("problem") or "").strip() for r in rows]
    problems = [p for p in problems if p][: args.max_problems]
    if not problems:
        raise ValueError("No valid problems to evaluate.")

    adapter_paths = [p.strip() for p in args.adapters.split(",") if p.strip()]
    reports = []
    for ap in adapter_paths:
        rep = evaluate(
            project_root=project_root,
            model_name=args.model_name,
            adapter_path=ap,
            problems=problems,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n_plans=args.n_plans,
            m_drafts=args.m_drafts,
            refine_rounds=args.refine_rounds,
            refine_top_k=args.refine_top_k,
        )
        reports.append(rep)
        print(
            f"{ap} -> avg_score={rep['avg_score']:.4f}, pass_rate={rep['pass_rate']:.4f}, n={rep['n_problems']}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"compare_{stamp}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "max_problems": len(problems),
        "reports": reports,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved comparison: {out_path}")


if __name__ == "__main__":
    main()
