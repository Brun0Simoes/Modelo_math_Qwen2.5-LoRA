import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def ensure_inside_root(path: Path, project_root: Path) -> None:
    abs_path = path.resolve()
    abs_root = project_root.resolve()
    if not str(abs_path).lower().startswith(str(abs_root).lower()):
        raise ValueError(f"Path must be inside project root: {path}")


def load_problem_text(args: argparse.Namespace) -> str:
    if args.problem_text:
        return args.problem_text.strip()
    if args.problem_file:
        return Path(args.problem_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide --problem-text or --problem-file.")


def to_markdown(run_data: dict) -> str:
    best = run_data["best_candidate"]
    parts = [
        "# Competitive Solver Run",
        "",
        "## Problem",
        run_data["problem"]["raw_text"],
        "",
        "## Parsed",
        f"- Domain: `{run_data['problem']['domain']}`",
        f"- Objective: {run_data['problem']['objective']}",
        "",
        "## Best Candidate",
        f"- Passed: `{best['verifier']['passed']}`",
        f"- Score: `{best['score']:.3f}`",
        f"- Plan: `{best['plan_name']}`",
        "",
        "### Draft",
        "```text",
        best["draft"],
        "```",
        "",
        "### Verifier Issues",
    ]
    issues = best["verifier"].get("issues", [])
    if not issues:
        parts.append("- none")
    else:
        for issue in issues:
            parts.append(f"- {issue}")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument("--problem-text", default="")
    parser.add_argument("--problem-file", default="")
    parser.add_argument("--output-dir", default="E:\\IA_matematica\\outputs\\eval\\competitive_solver")
    parser.add_argument("--backend", choices=["heuristic", "transformers"], default="heuristic")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter directory for transformers backend")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-plans", type=int, default=6)
    parser.add_argument("--m-drafts", type=int, default=3)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--refine-top-k", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    ensure_inside_root(output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from olympiad_system import (
        CandidateVerifier,
        CompetitiveSolver,
        HeuristicGenerator,
        SearchSettings,
        ToolSandbox,
        TransformersGenerator,
    )

    problem_text = load_problem_text(args)
    tools = ToolSandbox()
    verifier = CandidateVerifier(tools)

    if args.backend == "heuristic":
        generator = HeuristicGenerator(tools)
    else:
        generator = TransformersGenerator(
            model_name=args.model_name,
            adapter_path=(args.adapter_path.strip() or None),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    settings = SearchSettings(
        n_plans=args.n_plans,
        m_drafts=args.m_drafts,
        refine_rounds=args.refine_rounds,
        refine_top_k=args.refine_top_k,
    )
    solver = CompetitiveSolver(generator=generator, verifier=verifier, settings=settings)
    run = solver.solve(problem_text)
    run_data = run.to_dict()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"run_{stamp}.json"
    md_path = output_dir / f"run_{stamp}.md"
    json_path.write_text(json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(run_data), encoding="utf-8")

    best = run.best_candidate
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Best score={best.score:.3f}, passed={best.verifier.passed}, plan={best.plan_name}")


if __name__ == "__main__":
    main()
