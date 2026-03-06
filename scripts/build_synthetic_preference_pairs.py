import argparse
import json
import os
import random
import re
from pathlib import Path


def ensure_inside_root(path: Path, project_root: Path) -> None:
    if not str(path.resolve()).lower().startswith(str(project_root.resolve()).lower()):
        raise ValueError(f"Path must stay inside project root: {path}")


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def flip_inequalities(text: str) -> str:
    # Keep replacements deterministic and simple.
    out = text
    out = out.replace(">=", "__TMP_GE__")
    out = out.replace("<=", "__TMP_LE__")
    out = out.replace(">", "<")
    out = out.replace("<", ">")
    out = out.replace("__TMP_GE__", "<=")
    out = out.replace("__TMP_LE__", ">=")
    return out


def nudge_last_number(text: str, rng: random.Random) -> str:
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
    if not matches:
        return text
    m = matches[-1]
    token = m.group(0)
    try:
        if "." in token:
            value = float(token)
            delta = rng.choice([0.5, 1.0, -0.5, -1.0])
            new_token = f"{value + delta:.3f}".rstrip("0").rstrip(".")
        else:
            value = int(token)
            delta = rng.choice([1, 2, -1, -2])
            new_token = str(value + delta)
    except Exception:
        return text
    return text[: m.start()] + new_token + text[m.end() :]


def crop_tail(text: str, ratio: float) -> str:
    if len(text) < 80:
        return text
    cut = max(40, int(len(text) * ratio))
    return text[:cut].rstrip()


def make_negative(solution: str, rng: random.Random) -> str:
    candidates = []
    s = solution.strip()
    if not s:
        return ""

    # 1) Truncate proof and force a likely-wrong final statement.
    c1 = crop_tail(s, rng.uniform(0.45, 0.75))
    c1 += "\n\nFinal answer: This conclusion follows directly."
    candidates.append(c1)

    # 2) Flip inequalities.
    c2 = flip_inequalities(s)
    if c2 != s:
        candidates.append(c2)

    # 3) Modify last numeric token.
    c3 = nudge_last_number(s, rng)
    if c3 != s:
        candidates.append(c3)

    # 4) Remove lines likely to carry key constraints.
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 4:
        kept = [ln for i, ln in enumerate(lines) if i % 3 != 1]
        c4 = "\n".join(kept)
        if c4 and c4 != s:
            candidates.append(c4)

    # Pick the first meaningfully different option.
    for cand in candidates:
        cand = normalize_spaces(cand)
        if cand and cand != normalize_spaces(s):
            return cand
    return normalize_spaces(s) + "\nFinal answer: uncertain."


def first_nonempty(row: dict, keys: list[str]) -> str:
    for k in keys:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def load_rows(in_file: Path):
    with in_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", default=os.environ.get("PROJECT_ROOT", "E:\\IA_matematica")
    )
    parser.add_argument("--in-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--max-pairs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-solution-chars", type=int, default=80)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    in_file = Path(args.in_file)
    out_file = Path(args.out_file)
    ensure_inside_root(in_file, project_root)
    ensure_inside_root(out_file, project_root)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    pairs = 0
    seen = set()
    prompt_keys = ["prompt", "problem", "question", "input", "instruction"]
    solution_keys = ["solution", "response", "answer", "output"]

    with out_file.open("w", encoding="utf-8") as w:
        for row in load_rows(in_file):
            if pairs >= args.max_pairs:
                break
            prompt = first_nonempty(row, prompt_keys)
            chosen = first_nonempty(row, solution_keys)
            if not prompt or not chosen or len(chosen) < args.min_solution_chars:
                continue

            rejected = make_negative(chosen, rng)
            if not rejected or rejected.strip() == chosen.strip():
                continue

            key = (prompt.strip(), chosen.strip(), rejected.strip())
            if key in seen:
                continue
            seen.add(key)

            payload = {
                "prompt": prompt.strip(),
                "chosen": chosen.strip(),
                "rejected": rejected.strip(),
                "meta": {"source": "synthetic_corruption"},
            }
            w.write(json.dumps(payload, ensure_ascii=False) + "\n")
            pairs += 1

    print(f"Wrote {pairs} synthetic preference pairs -> {out_file}")


if __name__ == "__main__":
    main()
