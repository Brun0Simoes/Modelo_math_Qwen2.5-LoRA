import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class StageSpec:
    name: str
    sources: List[str]


STAGES = [
    StageSpec(
        name="stage1_foundations",
        sources=["deepmind_math_qa_train", "gsm8k_train", "mathqa_train", "openwebmath_train"],
    ),
    StageSpec(name="stage2_reasoning", sources=["gsm8k_train", "mathqa_train", "math_benchmark_train"]),
    StageSpec(name="stage3_competition", sources=["math_benchmark_train", "olympiadbench_en_train", "putnambench_train"]),
    StageSpec(name="stage4_formal_bridge", sources=["putnambench_train", "minif2f_test", "proofnet_lean4_valid"]),
]


def ensure_inside_root(path: Path, project_root: Path) -> None:
    if not str(path.resolve()).lower().startswith(str(project_root.resolve()).lower()):
        raise ValueError(f"Path must stay inside project root: {path}")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_pretrain_text(text: str) -> tuple[str, str]:
    cleaned = " ".join(text.split())
    if len(cleaned) < 240:
        return "", ""
    rough = int(len(cleaned) * 0.35)
    left = max(120, rough - 80)
    right = min(len(cleaned) - 120, rough + 80)
    split_idx = rough
    for i in range(rough, right):
        if cleaned[i] == " ":
            split_idx = i
            break
    else:
        for i in range(rough, left, -1):
            if cleaned[i] == " ":
                split_idx = i
                break
    prompt = cleaned[:split_idx].strip()
    continuation = cleaned[split_idx:].strip()
    if len(prompt) < 100 or len(continuation) < 100:
        return "", ""
    return prompt, continuation


def to_train_row(row: dict) -> dict | None:
    problem = (row.get("problem") or "").strip()
    solution = (row.get("solution") or "").strip()
    final_answer = (row.get("final_answer") or "").strip()
    task_type = (row.get("task_type") or "").strip()
    if not problem:
        return None
    if not solution:
        if final_answer:
            solution = f"Final answer: {final_answer}"
        elif task_type == "pretrain":
            prefix, continuation = split_pretrain_text(problem)
            if not prefix or not continuation:
                return None
            problem = (
                "Continue the following mathematical exposition with coherent steps:\n"
                f"{prefix}"
            )
            solution = continuation
        else:
            return None
    return {
        "problem": problem,
        "solution": solution,
        "source_dataset": row.get("source_dataset", ""),
        "split": row.get("split", ""),
        "task_type": row.get("task_type", ""),
        "license": row.get("license", ""),
    }


def parse_source_caps(values: List[str]) -> Dict[str, int]:
    caps: Dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --source-cap format: {item}. Expected source=count.")
        source, count_str = item.split("=", 1)
        source = source.strip()
        count_str = count_str.strip()
        if not source:
            raise ValueError(f"Invalid source name in --source-cap: {item}")
        count = int(count_str)
        if count <= 0:
            raise ValueError(f"Source cap must be > 0 for {source}")
        caps[source] = count
    return caps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument("--open-dir", default="E:\\IA_matematica\\data\\processed\\open")
    parser.add_argument("--out-dir", default="E:\\IA_matematica\\data\\processed\\curriculum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-source", type=int, default=20000)
    parser.add_argument(
        "--source-cap",
        action="append",
        default=[],
        help="Per-source cap override. Format: source_name=count. Can be repeated.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    open_dir = Path(args.open_dir)
    out_dir = Path(args.out_dir)
    ensure_inside_root(open_dir, project_root)
    ensure_inside_root(out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    source_caps = parse_source_caps(args.source_cap)
    source_files: Dict[str, Path] = {}
    for p in open_dir.glob("*.jsonl"):
        source_files[p.stem] = p

    manifest = {"stages": []}
    for stage in STAGES:
        stage_rows: List[dict] = []
        source_counts = {}
        for source in stage.sources:
            if source not in source_files:
                source_counts[source] = 0
                continue
            raw_rows = read_jsonl(source_files[source])
            rng.shuffle(raw_rows)
            source_limit = source_caps.get(source, args.max_per_source)
            count = 0
            for row in raw_rows:
                if count >= source_limit:
                    break
                train_row = to_train_row(row)
                if train_row is None:
                    continue
                stage_rows.append(train_row)
                count += 1
            source_counts[source] = count
        rng.shuffle(stage_rows)

        out_path = out_dir / f"{stage.name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in stage_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(stage_rows)} rows -> {out_path}")
        manifest["stages"].append(
            {
                "name": stage.name,
                "path": str(out_path),
                "rows": len(stage_rows),
                "source_counts": source_counts,
                "source_caps": {s: source_caps.get(s, args.max_per_source) for s in stage.sources},
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
