import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Tuple


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


def has_too_much_draw_noise(text: str) -> bool:
    lowered = text.lower()
    hits = sum(lowered.count(p.lower()) for p in DRAW_NOISE_PATTERNS)
    return hits >= 4


def clean_pair(
    row: dict,
    min_prompt_chars: int,
    min_response_chars: int,
    max_response_chars: int,
) -> dict | None:
    prompt = normalize_spaces(str(row.get("prompt") or ""))
    chosen = normalize_spaces(str(row.get("chosen") or ""))
    rejected = normalize_spaces(str(row.get("rejected") or ""))
    if not prompt or not chosen or not rejected:
        return None
    if len(prompt) < min_prompt_chars:
        return None
    if len(chosen) < min_response_chars or len(rejected) < min_response_chars:
        return None
    if len(chosen) > max_response_chars or len(rejected) > max_response_chars:
        return None
    if chosen == rejected:
        return None
    if has_too_much_draw_noise(chosen) and has_too_much_draw_noise(rejected):
        return None

    meta = row.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": meta,
    }


def dedup_pairs(rows: List[dict]) -> List[dict]:
    out = []
    seen: set[Tuple[str, str, str]] = set()
    for row in rows:
        key = (
            re.sub(r"\s+", " ", row["prompt"]).strip(),
            re.sub(r"\s+", " ", row["chosen"]).strip(),
            re.sub(r"\s+", " ", row["rejected"]).strip(),
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
        "--real-file",
        default="E:\\IA_matematica\\data\\processed\\preference_pairs_all_plus_synth_4k_dedup.jsonl",
    )
    parser.add_argument(
        "--synthetic-file",
        default="E:\\IA_matematica\\data\\processed\\preference_pairs_synth_stage9_6k.jsonl",
    )
    parser.add_argument(
        "--out-file",
        default="E:\\IA_matematica\\data\\processed\\preference_pairs_stage9_balanced_10k.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-total", type=int, default=10000)
    parser.add_argument("--max-synth-ratio", type=float, default=0.45)
    parser.add_argument("--min-prompt-chars", type=int, default=24)
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--max-response-chars", type=int, default=3400)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    real_file = Path(args.real_file)
    synthetic_file = Path(args.synthetic_file)
    out_file = Path(args.out_file)

    for p in [real_file, synthetic_file, out_file]:
        ensure_inside_root(p, project_root)
    if not real_file.exists():
        raise FileNotFoundError(f"Real preference file not found: {real_file}")
    if not synthetic_file.exists():
        raise FileNotFoundError(f"Synthetic preference file not found: {synthetic_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    real_rows = []
    for row in read_jsonl(real_file):
        cleaned = clean_pair(
            row=row,
            min_prompt_chars=args.min_prompt_chars,
            min_response_chars=args.min_response_chars,
            max_response_chars=args.max_response_chars,
        )
        if cleaned is None:
            continue
        meta = dict(cleaned.get("meta") or {})
        meta["source"] = meta.get("source") or "real_preference"
        cleaned["meta"] = meta
        real_rows.append(cleaned)

    synth_rows = []
    for row in read_jsonl(synthetic_file):
        cleaned = clean_pair(
            row=row,
            min_prompt_chars=args.min_prompt_chars,
            min_response_chars=args.min_response_chars,
            max_response_chars=args.max_response_chars,
        )
        if cleaned is None:
            continue
        meta = dict(cleaned.get("meta") or {})
        meta["source"] = "synthetic_corruption"
        cleaned["meta"] = meta
        synth_rows.append(cleaned)

    real_rows = dedup_pairs(real_rows)
    synth_rows = dedup_pairs(synth_rows)
    rng.shuffle(real_rows)
    rng.shuffle(synth_rows)

    target_total = max(1, args.target_total)
    max_synth = int(target_total * args.max_synth_ratio)
    max_real = target_total - max_synth

    selected_real = real_rows[: min(len(real_rows), max_real)]
    remaining = target_total - len(selected_real)
    selected_synth = synth_rows[: min(len(synth_rows), max_synth, remaining)]

    selected = dedup_pairs(selected_real + selected_synth)
    if len(selected) < target_total:
        # Backfill with any remaining real rows first, then synthetic.
        used_keys = {
            (
                re.sub(r"\s+", " ", x["prompt"]).strip(),
                re.sub(r"\s+", " ", x["chosen"]).strip(),
                re.sub(r"\s+", " ", x["rejected"]).strip(),
            )
            for x in selected
        }
        for pool in (real_rows[len(selected_real) :], synth_rows[len(selected_synth) :]):
            for row in pool:
                if len(selected) >= target_total:
                    break
                key = (
                    re.sub(r"\s+", " ", row["prompt"]).strip(),
                    re.sub(r"\s+", " ", row["chosen"]).strip(),
                    re.sub(r"\s+", " ", row["rejected"]).strip(),
                )
                if key in used_keys:
                    continue
                used_keys.add(key)
                selected.append(row)

    rng.shuffle(selected)

    with out_file.open("w", encoding="utf-8") as w:
        for row in selected:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    synth_count = sum(1 for row in selected if row.get("meta", {}).get("source") == "synthetic_corruption")
    real_count = len(selected) - synth_count
    print(f"Real rows available: {len(real_rows)}")
    print(f"Synthetic rows available: {len(synth_rows)}")
    print(f"Wrote {len(selected)} rows -> {out_file}")
    print(f"Composition: real={real_count}, synthetic={synth_count}, synth_ratio={synth_count / max(1, len(selected)):.4f}")


if __name__ == "__main__":
    main()
