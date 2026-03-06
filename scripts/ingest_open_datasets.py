import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


def ensure_inside_root(path: Path, project_root: Path) -> None:
    if not str(path.resolve()).lower().startswith(str(project_root.resolve()).lower()):
        raise ValueError(f"Path must stay inside project root: {path}")


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [normalize_value(v) for v in value if normalize_value(v)]
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def pick_field(row: Dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in row:
            val = normalize_value(row[key])
            if val:
                return val
    return ""


def extract_gsm8k_final_answer(solution: str) -> str:
    marker = "####"
    if marker in solution:
        return solution.split(marker, 1)[1].strip()
    return ""


def iter_rows(ds, max_rows: Optional[int]):
    count = 0
    for row in ds:
        yield row
        count += 1
        if max_rows is not None and count >= max_rows:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica"))
    parser.add_argument("--config", default="E:\\IA_matematica\\configs\\open_datasets.json")
    parser.add_argument("--datasets", default="all", help="Comma-separated names or 'all'")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HF streaming mode (default: true)",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Override row limit per split (0 = spec default)")
    parser.add_argument("--out-dir", default="E:\\IA_matematica\\data\\processed\\open")
    parser.add_argument("--raw-cache-dir", default="E:\\IA_matematica\\data\\raw\\hf_cache")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    raw_cache = Path(args.raw_cache_dir)
    ensure_inside_root(config_path, project_root)
    ensure_inside_root(out_dir, project_root)
    ensure_inside_root(raw_cache, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache.mkdir(parents=True, exist_ok=True)

    config = read_config(config_path)
    specs = config.get("datasets", [])
    if not specs:
        raise ValueError("No dataset specs in config.")

    selected = None
    if args.datasets.strip().lower() != "all":
        selected = {name.strip() for name in args.datasets.split(",") if name.strip()}

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "streaming": bool(args.streaming),
        "datasets": [],
    }

    for spec in specs:
        name = spec["name"]
        if selected is not None and name not in selected:
            continue

        hf_path = spec["hf_path"]
        hf_config = spec.get("hf_config")
        task_type = spec.get("task_type", "sft")
        license_name = spec.get("license", "unknown")
        fields = spec.get("fields", {})
        splits = spec.get("splits", ["train"])
        default_max_rows = spec.get("default_max_rows")
        ds_summary = {"name": name, "files": []}

        print(f"Processing dataset: {name} ({hf_path})")
        for split in splits:
            load_kwargs: Dict[str, Any] = {
                "path": hf_path,
                "split": split,
                "streaming": bool(args.streaming),
                "cache_dir": str(raw_cache),
            }
            if hf_config:
                load_kwargs["name"] = hf_config
            ds = load_dataset(**load_kwargs)
            out_file = out_dir / f"{name}_{split}.jsonl"
            limit = args.max_rows if args.max_rows > 0 else default_max_rows
            count = 0
            with out_file.open("w", encoding="utf-8") as f:
                for row in iter_rows(ds, limit):
                    problem = pick_field(row, fields.get("problem", []))
                    solution = pick_field(row, fields.get("solution", []))
                    final_answer = pick_field(row, fields.get("final_answer", []))
                    if name == "gsm8k" and solution and not final_answer:
                        final_answer = extract_gsm8k_final_answer(solution)
                    record = {
                        "source_dataset": name,
                        "hf_path": hf_path,
                        "split": split,
                        "task_type": task_type,
                        "license": license_name,
                        "id": pick_field(row, fields.get("id", [])),
                        "problem": problem,
                        "solution": solution,
                        "final_answer": final_answer,
                        "metadata": {
                            "available_columns": sorted(list(row.keys())),
                        },
                    }
                    if not record["problem"]:
                        continue
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            print(f"  wrote {count} rows -> {out_file}")
            ds_summary["files"].append({"split": split, "path": str(out_file), "rows": count})
        manifest["datasets"].append(ds_summary)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
