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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", default="E:\\IA_matematica\\outputs\\eval\\competitive_solver")
    parser.add_argument("--backend", choices=["heuristic", "transformers"], default="heuristic")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter directory for transformers backend")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-plans", type=int, default=6)
    parser.add_argument("--m-drafts", type=int, default=3)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--refine-top-k", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    ensure_inside_root(input_path, project_root)
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

    rows = read_jsonl(input_path)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_problems]

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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = 0
    for i, row in enumerate(rows):
        problem = (row.get("problem") or "").strip()
        if not problem:
            continue
        run = solver.solve(problem)
        data = run.to_dict()
        data["source"] = {
            "index": i,
            "source_dataset": row.get("source_dataset", ""),
            "split": row.get("split", ""),
        }
        out_path = output_dir / f"run_{stamp}_{i:04d}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
        if written % 10 == 0:
            print(f"processed {written}/{len(rows)}")
    print(f"Wrote {written} run files to {output_dir}")


if __name__ == "__main__":
    main()
