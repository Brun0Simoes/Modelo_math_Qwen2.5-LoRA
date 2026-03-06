import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FINAL_PATTERNS = [
    r"final answer\s*:\s*(.+)",
    r"answer\s*:\s*(.+)",
    r"resposta final\s*:\s*(.+)",
]

DRAW_NOISE_PATTERNS = [
    "draw(",
    "label(",
    "tikzpicture",
    "[asy]",
    "\\begin{asy}",
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


def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_heavy_blocks(text: str) -> str:
    text = re.sub(r"\[asy\].*?\[/asy\]", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"```(?:asy|python|text)?\n.*?```",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return normalize_spaces(text)


def extract_final_answer(solution: str) -> str:
    if not solution:
        return ""
    for pat in FINAL_PATTERNS:
        m = re.search(pat, solution, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().splitlines()[0].strip()
    boxed = re.findall(r"\\boxed\{([^{}]{1,240})\}", solution)
    if boxed:
        return boxed[-1].strip()
    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    if len(tail) <= 180 and len(tail.split()) <= 24:
        return tail
    return ""


def has_too_much_draw_noise(text: str) -> bool:
    lowered = text.lower()
    hits = sum(lowered.count(p.lower()) for p in DRAW_NOISE_PATTERNS)
    return hits >= 3


def source_from_row(row: dict, fallback: str) -> str:
    src = str(row.get("source_dataset") or "").strip()
    if src:
        return src
    meta = row.get("meta")
    if isinstance(meta, dict):
        msrc = str(meta.get("source_dataset") or meta.get("source") or "").strip()
        if msrc:
            return msrc
    return fallback


def to_train_row(
    row: dict,
    fallback_source: str,
    min_problem_chars: int,
    max_problem_chars: int,
    min_solution_chars: int,
    max_solution_chars: int,
    force_final_answer: bool,
) -> dict | None:
    problem = normalize_spaces(str(row.get("problem") or row.get("prompt") or ""))
    solution = normalize_spaces(str(row.get("solution") or row.get("response") or ""))

    # Accept tool-trace style rows as supervised samples.
    if not problem and isinstance(row.get("prompt"), str):
        problem = normalize_spaces(row["prompt"])
    if not solution and row.get("tool_name"):
        before = normalize_spaces(str(row.get("assistant_before_tool") or ""))
        tool = normalize_spaces(str(row.get("tool_name") or ""))
        tool_out = normalize_spaces(str(row.get("tool_output") or ""))
        after = normalize_spaces(str(row.get("assistant_after_tool") or ""))
        parts = [p for p in [before, f"[{tool}] {tool_out}", after] if p]
        solution = "\n".join(parts).strip()

    if not problem or not solution:
        return None

    problem = strip_heavy_blocks(problem)
    solution = strip_heavy_blocks(solution)
    if not problem or not solution:
        return None

    if has_too_much_draw_noise(solution):
        return None

    if not (min_problem_chars <= len(problem) <= max_problem_chars):
        return None
    if not (min_solution_chars <= len(solution) <= max_solution_chars):
        return None

    if force_final_answer and not re.search(r"(?im)^\s*final answer\s*:", solution):
        final_answer = extract_final_answer(solution)
        if final_answer:
            solution = f"{solution}\n\nFinal answer: {final_answer}"

    source = source_from_row(row, fallback=fallback_source)
    out = {
        "problem": problem,
        "solution": solution,
        "source_dataset": source,
        "split": str(row.get("split") or "train"),
        "task_type": str(row.get("task_type") or "sft"),
        "license": str(row.get("license") or ""),
    }
    return out


def parse_source_caps(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --source-cap format: {item}. Expected source=count.")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Empty source name in --source-cap: {item}")
        cap = int(value)
        if cap <= 0:
            raise ValueError(f"Cap must be > 0 for source '{name}'")
        out[name] = cap
    return out


def dedup_rows(rows: List[dict]) -> List[dict]:
    seen: set[Tuple[str, str]] = set()
    out = []
    for row in rows:
        key = (
            re.sub(r"\s+", " ", row["problem"]).strip(),
            re.sub(r"\s+", " ", row["solution"]).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"),
    )
    parser.add_argument(
        "--in-files",
        default=(
            "E:\\IA_matematica\\data\\processed\\curriculum\\stage1plus234_competitive_boost.jsonl,"
            "E:\\IA_matematica\\data\\processed\\curriculum\\stage8_distill_large_mix_v2.jsonl"
        ),
        help="Comma-separated JSONL input files.",
    )
    parser.add_argument(
        "--tool-traces-file",
        default="E:\\IA_matematica\\data\\processed\\tool_traces.jsonl",
    )
    parser.add_argument(
        "--out-file",
        default="E:\\IA_matematica\\data\\processed\\curriculum\\stage9_final_heavy_mix.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-problem-chars", type=int, default=20)
    parser.add_argument("--max-problem-chars", type=int, default=5000)
    parser.add_argument("--min-solution-chars", type=int, default=60)
    parser.add_argument("--max-solution-chars", type=int, default=3800)
    parser.add_argument("--default-cap", type=int, default=20000)
    parser.add_argument(
        "--source-cap",
        action="append",
        default=[],
        help="Per-source cap override. Format: source_name=count. Can be repeated.",
    )
    parser.add_argument("--max-total", type=int, default=0, help="0 means no global cap.")
    parser.add_argument(
        "--force-final-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    out_file = Path(args.out_file)
    ensure_inside_root(out_file, project_root)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    input_files = [Path(p.strip()) for p in args.in_files.split(",") if p.strip()]
    for p in input_files:
        ensure_inside_root(p, project_root)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    tool_traces_file = Path(args.tool_traces_file)
    include_tool_traces = tool_traces_file.exists()
    if include_tool_traces:
        ensure_inside_root(tool_traces_file, project_root)

    caps = parse_source_caps(args.source_cap)
    rng = random.Random(args.seed)

    bucket: Dict[str, List[dict]] = defaultdict(list)
    read_total = 0
    for path in input_files:
        fallback_source = path.stem
        for row in read_jsonl(path):
            read_total += 1
            item = to_train_row(
                row=row,
                fallback_source=fallback_source,
                min_problem_chars=args.min_problem_chars,
                max_problem_chars=args.max_problem_chars,
                min_solution_chars=args.min_solution_chars,
                max_solution_chars=args.max_solution_chars,
                force_final_answer=args.force_final_answer,
            )
            if item is None:
                continue
            bucket[item["source_dataset"]].append(item)

    if include_tool_traces:
        for row in read_jsonl(tool_traces_file):
            read_total += 1
            item = to_train_row(
                row=row,
                fallback_source="tool_traces",
                min_problem_chars=args.min_problem_chars,
                max_problem_chars=args.max_problem_chars,
                min_solution_chars=max(40, args.min_solution_chars // 2),
                max_solution_chars=args.max_solution_chars,
                force_final_answer=False,
            )
            if item is None:
                continue
            item["source_dataset"] = "tool_traces"
            bucket[item["source_dataset"]].append(item)

    selected: List[dict] = []
    source_counts: Dict[str, int] = {}
    for source, rows in bucket.items():
        rng.shuffle(rows)
        cap = caps.get(source, args.default_cap)
        picked = rows[:cap]
        source_counts[source] = len(picked)
        selected.extend(picked)

    selected = dedup_rows(selected)
    rng.shuffle(selected)
    if args.max_total > 0 and len(selected) > args.max_total:
        selected = selected[: args.max_total]

    with out_file.open("w", encoding="utf-8") as w:
        for row in selected:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Rows read: {read_total}")
    print(f"Rows written: {len(selected)} -> {out_file}")
    print("Source counts after cap:")
    for source, count in sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  - {source}: {count}")


if __name__ == "__main__":
    main()
