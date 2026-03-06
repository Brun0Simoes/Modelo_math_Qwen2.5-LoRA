from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import List, Optional

import sympy as sp

from .schemas import ParsedProblem, StrategyPlan
from .tools import ToolSandbox, parse_math_expr


@dataclass
class Draft:
    text: str
    tool_logs: List[str]


def _extract_options(problem_text: str) -> dict[str, float]:
    options: dict[str, float] = {}
    for line in str(problem_text or "").splitlines():
        m = re.match(
            r"^\s*([A-Ea-e])(?:\s*[).:\-]\s*|\s+)([0-9][0-9\s.,]*)\s*$",
            line,
        )
        if not m:
            continue
        raw = m.group(2).strip().replace(" ", "")
        raw = raw.replace(",", ".")
        try:
            options[m.group(1).upper()] = float(raw)
        except Exception:
            continue
    return options


def _is_proof_like_problem(problem_text: str) -> bool:
    lower = str(problem_text or "").lower()
    proof_markers = ["prove", "show", "mostre", "demonstre", "prove que"]
    return any(m in lower for m in proof_markers)


def _parse_number_token(token: str) -> Optional[float]:
    raw = str(token or "").strip().replace(" ", "")
    raw = raw.replace(",", ".")
    try:
        return float(raw)
    except Exception:
        return None


def _match_option_letter(text: str, options: dict[str, float]) -> Optional[str]:
    if not options:
        return None
    t = str(text or "")
    m = re.search(r"\b(?:option|alternativa|opcao)\s*([A-Ea-e])\b", t, flags=re.IGNORECASE)
    if m is None:
        m = re.match(r"^\s*([A-Ea-e])\s*(?:[\).:\-]|\(|$)", t.strip())
    if m:
        letter = m.group(1).upper()
        if letter in options:
            return letter
    for tok in re.findall(r"-?\d+(?:[.,]\d+)?", t):
        num = _parse_number_token(tok)
        if num is None:
            continue
        for letter, value in options.items():
            if abs(num - value) < 1e-6:
                return letter
    return None


def _format_option_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def _canonical_option_answer(letter: str, options: dict[str, float]) -> str:
    value = options.get(letter)
    if value is None:
        return letter
    return f"{letter} ({_format_option_value(value)})"


def _extract_final_answer(text: str) -> Optional[str]:
    patterns = [
        r"final answer\s*:\s*(.+)",
        r"answer\s*:\s*(.+)",
        r"resposta final\s*:\s*(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            return ans.splitlines()[0].strip()
    boxed = re.findall(r"\\boxed\{([^{}]{1,220})\}", text)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        if len(last) <= 200 and len(last.split()) <= 36:
            return last
    return None


def _solve_simple_system(problem: ParsedProblem) -> Optional[str]:
    # Quick deterministic helper for equation-only prompts.
    relations = []
    for line in problem.raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line and len(line) <= 120:
            relations.append(line.replace("==", "="))
    if not relations:
        return None
    eqs = []
    symbols = set()
    try:
        for rel in relations:
            if "=" not in rel:
                continue
            left, right = rel.split("=", 1)
            lhs = parse_math_expr(left.strip())
            rhs = parse_math_expr(right.strip())
            eqs.append(sp.Eq(lhs, rhs))
            symbols |= lhs.free_symbols
            symbols |= rhs.free_symbols
        if not eqs or not symbols:
            return None
        ordered = sorted(list(symbols), key=lambda x: str(x))
        sol = sp.solve(eqs, ordered, dict=True)
        if not sol:
            return None
        picked = sol[0]
        parts = [f"{str(k)} = {sp.simplify(v)}" for k, v in picked.items()]
        return ", ".join(parts)
    except Exception:
        return None


class HeuristicGenerator:
    def __init__(self, tools: ToolSandbox) -> None:
        self.tools = tools
        self._counter = itertools.count(1)

    def generate(
        self,
        problem: ParsedProblem,
        plan: StrategyPlan,
        n: int,
        feedback: Optional[str] = None,
    ) -> List[Draft]:
        out: List[Draft] = []
        for _ in range(max(1, n)):
            step_id = next(self._counter)
            tool_logs: List[str] = []
            quick_answer = _solve_simple_system(problem)
            if quick_answer:
                tool_logs.append("sympy.solve_system -> success")
            else:
                tool_logs.append("sympy.solve_system -> no direct closed-form extraction")

            fb_line = ""
            if feedback:
                fb_line = f"Critic feedback to address: {feedback}"
            final_answer = quick_answer or "pending verification; requires deeper derivation."
            text = "\n".join(
                [
                    f"Strategy: {plan.name}",
                    f"Rationale: {plan.rationale}",
                    f"Attempt #{step_id}",
                    f"Problem objective: {problem.objective or 'derive target statement'}",
                    "Step 1: Restate hypotheses and isolate controllable variables.",
                    "Step 2: Apply the selected strategy to produce an intermediate relation.",
                    "Step 3: Validate critical equalities/constraints with CAS or numeric checks.",
                    fb_line,
                    f"Final answer: {final_answer}",
                ]
            ).strip()
            out.append(Draft(text=text, tool_logs=tool_logs))
        return out


class TransformersGenerator:
    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        load_in_4bit: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.load_in_4bit = load_in_4bit

        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = {"": 0}
            self._input_device = torch.device("cuda:0")
        else:
            torch_dtype = torch.float32
            device_map = None
            self._input_device = torch.device("cpu")

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        # 4-bit quantization for large models (7B+)
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        if adapter_path:
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(
                adapter_path,
                **model_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        problem: ParsedProblem,
        plan: StrategyPlan,
        n: int,
        feedback: Optional[str] = None,
    ) -> List[Draft]:
        import torch

        options = _extract_options(problem.raw_text)
        objective_mcq = bool(options) and not _is_proof_like_problem(problem.raw_text)
        fb = f"\nCritic feedback: {feedback}\n" if feedback else ""
        if objective_mcq:
            option_lines = "\n".join(
                f"{letter}: {_format_option_value(value)}" for letter, value in sorted(options.items())
            )
            prompt = (
                "You are solving a deterministic multiple-choice math question.\n"
                "Rules:\n"
                "1) Do not invent data or assumptions.\n"
                "2) Compute from the statement only.\n"
                "3) Keep reasoning concise (max 8 lines).\n"
                "4) Finish with exactly one line in this format:\n"
                "Final answer: <LETTER> (<NUMERIC_OPTION_VALUE>)\n\n"
                f"Domain: {problem.domain}\n"
                f"Strategy: {plan.name}\n"
                f"Rationale: {plan.rationale}\n"
                f"{plan.prompt_template}\n"
                f"{fb}\n"
                "Options:\n"
                f"{option_lines}\n\n"
                f"Problem:\n{problem.raw_text}\n\n"
                "Draft solution:\n"
            )
        else:
            prompt = (
                "You are solving a math olympiad problem. "
                "Write a rigorous but concise draft. "
                "Use explicit steps and finish with exactly one line: `Final answer: ...`.\n\n"
                f"Domain: {problem.domain}\n"
                f"Strategy: {plan.name}\n"
                f"Rationale: {plan.rationale}\n"
                f"{plan.prompt_template}\n"
                f"{fb}\n"
                f"Problem:\n{problem.raw_text}\n\n"
                "Draft solution:\n"
            )

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        )
        inputs = {k: v.to(self._input_device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        generation_budget = min(self.max_new_tokens, 256) if objective_mcq else self.max_new_tokens

        do_sample = self.temperature > 0 and not objective_mcq
        num_sequences = max(1, int(n))
        gen_kwargs = {
            "do_sample": do_sample,
            "max_length": prompt_len + generation_budget,
            "num_return_sequences": num_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.02 if objective_mcq else 1.05,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.92
        else:
            gen_kwargs["num_beams"] = max(3 if objective_mcq else 2, num_sequences)
            gen_kwargs["early_stopping"] = True

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)

        completions = generated[:, prompt_len:]
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)

        drafts = []
        for text in texts:
            text = str(text).strip()
            final_answer = _extract_final_answer(text)
            if objective_mcq:
                voted = _match_option_letter(final_answer or text, options)
                if voted is not None:
                    canonical = _canonical_option_answer(voted, options)
                    if re.search(r"(?im)^\s*final answer\s*:", text):
                        text = re.sub(
                            r"(?im)^\s*final answer\s*:.*$",
                            f"Final answer: {canonical}",
                            text,
                        ).strip()
                    else:
                        text = f"{text}\nFinal answer: {canonical}".strip()
                else:
                    if re.search(r"(?im)^\s*final answer\s*:", text):
                        text = re.sub(
                            r"(?im)^\s*final answer\s*:.*$",
                            "Final answer: unable to determine",
                            text,
                        ).strip()
                    else:
                        text = f"{text}\nFinal answer: unable to determine".strip()
            else:
                if final_answer and not re.search(r"(?im)^\s*final answer\s*:", text):
                    text = f"{text}\nFinal answer: {final_answer}".strip()
                elif not final_answer:
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    if lines:
                        guess = lines[-1]
                        if len(guess) <= 200:
                            text = f"{text}\nFinal answer: {guess}".strip()
                        else:
                            text = f"{text}\nFinal answer: unable to determine".strip()
                    else:
                        text = "Final answer: unable to determine"
            drafts.append(Draft(text=text, tool_logs=["llm.generate -> ok"]))
        return drafts
