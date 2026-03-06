import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FINAL_PATTERNS = [
    r"final answer\s*:\s*(.+)",
    r"answer\s*:\s*(.+)",
    r"resposta final\s*:\s*(.+)",
]


def ensure_inside_root(path: Path, project_root: Path) -> None:
    if not str(path.resolve()).lower().startswith(str(project_root.resolve()).lower()):
        raise ValueError(f"Path must stay inside project root: {path}")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def clean_text(text: str) -> str:
    # Remove common heavy blocks that harm SFT signal quality.
    text = re.sub(r"\[asy\].*?\[/asy\]", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"```(?:asy|python|text)?\n.*?```", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_final_answer(text: str) -> str:
    raw = text.strip()
    if not raw:
        return ""
    for pat in FINAL_PATTERNS:
        m = re.search(pat, raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().splitlines()[0].strip()
    boxed = re.findall(r"\\boxed\{([^{}]{1,240})\}", raw)
    if boxed:
        return boxed[-1].strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    if len(tail) <= 220:
        return tail
    return ""


def normalize_solution(text: str) -> str:
    sol = clean_text(text)
    if not sol:
        return ""
    if not re.search(r"(?im)^\s*final answer\s*:", sol):
        fa = extract_final_answer(sol)
        if fa:
            sol = f"{sol}\n\nFinal answer: {fa}".strip()
    return sol


def is_competition_like(problem: str, source: str) -> bool:
    s = source.lower()
    p = problem.lower()
    if any(k in s for k in ["olympiad", "putnam", "benchmark"]):
        return True
    if any(k in p for k in ["prove", "show that", "mostre", "inequality", "congruence", "triangle"]):
        return True
    return False


def read_run_candidates(
    runs_dir: Path,
    min_score: float,
    max_candidates_per_problem: int,
) -> List[dict]:
    out: List[dict] = []
    for p in sorted(runs_dir.glob("run_*.json")):
        try:
            run = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        problem = (run.get("problem") or {}).get("raw_text", "").strip()
        if not problem:
            continue
        cands = run.get("candidates", [])
        good = []
        for c in cands:
            score = float(c.get("score", 0.0))
            passed = bool((c.get("verifier") or {}).get("passed", False))
            if not passed or score < min_score:
                continue
            draft = normalize_solution(str(c.get("draft") or ""))
            if not draft:
                continue
            good.append((score, draft))
        good.sort(key=lambda t: t[0], reverse=True)
        for _, draft in good[:max_candidates_per_problem]:
            out.append(
                {
                    "problem": problem,
                    "solution": draft,
                    "source_dataset": "distill_stage8_high",
                    "split": "train",
                    "task_type": "sft",
                    "license": "derived-local",
                }
            )
    return out


def select_base_rows(
    base_rows: List[dict],
    total_samples: int,
    comp_ratio: float,
    rng: random.Random,
    min_solution_chars: int,
    max_solution_chars: int,
) -> List[dict]:
    comp: List[dict] = []
    other: List[dict] = []
    for r in base_rows:
        problem = str(r.get("problem") or "").strip()
        solution = normalize_solution(str(r.get("solution") or ""))
        if not problem or not solution:
            continue
        if not (min_solution_chars <= len(solution) <= max_solution_chars):
            continue
        row = {
            "problem": problem,
            "solution": solution,
            "source_dataset": r.get("source_dataset", ""),
            "split": r.get("split", "train"),
            "task_type": r.get("task_type", "sft"),
            "license": r.get("license", ""),
        }
        if is_competition_like(problem, str(r.get("source_dataset") or "")):
            comp.append(row)
        else:
            other.append(row)

    rng.shuffle(comp)
    rng.shuffle(other)
    target_comp = min(int(total_samples * comp_ratio), len(comp))
    target_other = min(total_samples - target_comp, len(other))

    selected = comp[:target_comp] + other[:target_other]
    rng.shuffle(selected)
    return selected


def dedup_rows(rows: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for r in rows:
        key = (
            re.sub(r"\s+", " ", str(r.get("problem") or "").strip()),
            re.sub(r"\s+", " ", str(r.get("solution") or "").strip()),
        )
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument(
        "--base-file",
        default="E:\\IA_matematica\\data\\processed\\curriculum\\stage1plus234_competitive_boost.jsonl",
    )
    parser.add_argument(
        "--runs-dir",
        default="E:\\IA_matematica\\outputs\\eval\\distill_stage8_high",
    )
    parser.add_argument(
        "--out-file",
        default="E:\\IA_matematica\\data\\processed\\curriculum\\stage8_distill_large_mix.jsonl",
    )
    parser.add_argument("--base-samples", type=int, default=50000)
    parser.add_argument("--comp-ratio", type=float, default=0.72)
    parser.add_argument("--min-solution-chars", type=int, default=80)
    parser.add_argument("--max-solution-chars", type=int, default=3000)
    parser.add_argument("--run-min-score", type=float, default=0.45)
    parser.add_argument("--run-max-cands-per-problem", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    base_file = Path(args.base_file)
    runs_dir = Path(args.runs_dir)
    out_file = Path(args.out_file)

    ensure_inside_root(base_file, project_root)
    ensure_inside_root(runs_dir, project_root)
    ensure_inside_root(out_file, project_root)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    base_rows = list(read_jsonl(base_file))
    rng.shuffle(base_rows)
    picked_base = select_base_rows(
        base_rows=base_rows,
        total_samples=args.base_samples,
        comp_ratio=args.comp_ratio,
        rng=rng,
        min_solution_chars=args.min_solution_chars,
        max_solution_chars=args.max_solution_chars,
    )
    distill_rows = read_run_candidates(
        runs_dir=runs_dir,
        min_score=args.run_min_score,
        max_candidates_per_problem=args.run_max_cands_per_problem,
    )

    merged = dedup_rows(distill_rows + picked_base)
    rng.shuffle(merged)

    with out_file.open("w", encoding="utf-8") as w:
        for r in merged:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Base rows selected: {len(picked_base)}")
    print(f"Distill rows selected: {len(distill_rows)}")
    print(f"Merged rows written: {len(merged)} -> {out_file}")


if __name__ == "__main__":
    main()
