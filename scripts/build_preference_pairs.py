import argparse
import json
import os
from pathlib import Path
from typing import List


def ensure_inside_root(path: Path, project_root: Path) -> None:
    if not str(path.resolve()).lower().startswith(str(project_root.resolve()).lower()):
        raise ValueError(f"Path must stay inside project root: {path}")


def load_runs(runs_dir: Path) -> List[dict]:
    runs = []
    for p in sorted(runs_dir.glob("run_*.json")):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument("--runs-dir", default="E:\\IA_matematica\\outputs\\eval\\competitive_solver")
    parser.add_argument("--out-file", default="E:\\IA_matematica\\data\\processed\\preference_pairs.jsonl")
    parser.add_argument("--min-score-gap", type=float, default=0.15)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    runs_dir = Path(args.runs_dir)
    out_file = Path(args.out_file)
    ensure_inside_root(runs_dir, project_root)
    ensure_inside_root(out_file, project_root)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_dir)
    total = 0
    with out_file.open("w", encoding="utf-8") as f:
        for run in runs:
            problem = run["problem"]["raw_text"]
            candidates = run.get("candidates", [])
            if len(candidates) < 2:
                continue
            ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
            chosen = ranked[0]
            rejected = ranked[-1]
            if chosen.get("score", 0.0) - rejected.get("score", 0.0) < args.min_score_gap:
                continue
            row = {
                "prompt": problem,
                "chosen": chosen["draft"],
                "rejected": rejected["draft"],
                "meta": {
                    "chosen_score": chosen.get("score"),
                    "rejected_score": rejected.get("score"),
                    "chosen_passed": chosen.get("verifier", {}).get("passed", False),
                    "rejected_passed": rejected.get("verifier", {}).get("passed", False),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
    print(f"Wrote {total} preference pairs -> {out_file}")


if __name__ == "__main__":
    main()
