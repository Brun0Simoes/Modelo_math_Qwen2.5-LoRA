import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import sys


def ensure_inside_root(path: Path, project_root: Path) -> None:
    if not str(path.resolve()).lower().startswith(str(project_root.resolve()).lower()):
        raise ValueError(f"Path must stay inside project root: {path}")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_equalities(text: str) -> List[Tuple[str, str]]:
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" not in line or len(line) > 140:
            continue
        if any(op in line for op in (">=", "<=", "=>", "->")):
            continue
        if line.count("=") != 1:
            continue
        left, right = line.split("=", 1)
        if re.search(r"[A-Za-z0-9\)\]]", left) and re.search(r"[A-Za-z0-9\(\[]", right):
            pairs.append((left.strip(), right.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="E:\\IA_matematica\\data\\processed\\tool_traces.jsonl")
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    ensure_inside_root(input_path, project_root)
    ensure_inside_root(output_path, project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from olympiad_system.tools import ToolSandbox

    tools = ToolSandbox()
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(read_jsonl(input_path)):
            if idx >= args.max_rows:
                break
            problem = (row.get("problem") or "").strip()
            solution = (row.get("solution") or "").strip()
            if not problem or not solution:
                continue
            equalities = extract_equalities(solution)
            for left, right in equalities[:4]:
                check = tools.check_identity(left, right, trials=8)
                trace = {
                    "problem": problem,
                    "assistant_before_tool": f"Verify whether `{left} = {right}` holds.",
                    "tool_name": "check_identity",
                    "tool_input": {"left": left, "right": right},
                    "tool_output": {"ok": check.ok, "value": check.value, "detail": check.detail},
                    "assistant_after_tool": (
                        "Equality check passed on random tests."
                        if check.ok
                        else "Equality check found a counterexample or parse failure."
                    ),
                    "source_dataset": row.get("source_dataset", ""),
                }
                out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                written += 1
    print(f"Wrote {written} tool traces -> {output_path}")


if __name__ == "__main__":
    main()
