from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import sympy as sp
import z3
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _safe_locals(expr: str) -> Dict[str, object]:
    allowed = {
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
        "exp": sp.exp,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    for tok in set(re.findall(r"\b[a-zA-Z_]\w*\b", expr)):
        if tok not in allowed:
            allowed[tok] = sp.Symbol(tok)
    return allowed


def parse_math_expr(expr: str) -> sp.Expr:
    return parse_expr(expr, local_dict=_safe_locals(expr), transformations=TRANSFORMATIONS)


def _split_relation(text: str) -> Tuple[str, str, str]:
    for op in ("==", ">=", "<=", "!=", "=", ">", "<"):
        if op in text:
            left, right = text.split(op, 1)
            return left.strip(), op, right.strip()
    raise ValueError(f"No relation operator found in: {text}")


def _sympy_to_z3(expr: sp.Expr, var_map: Dict[str, z3.ArithRef]) -> z3.ArithRef:
    if expr.is_Number:
        return z3.RealVal(str(expr))
    if expr.is_Symbol:
        name = str(expr)
        if name not in var_map:
            var_map[name] = z3.Real(name)
        return var_map[name]
    if expr.is_Add:
        out = z3.RealVal("0")
        for arg in expr.args:
            out = out + _sympy_to_z3(arg, var_map)
        return out
    if expr.is_Mul:
        out = z3.RealVal("1")
        for arg in expr.args:
            out = out * _sympy_to_z3(arg, var_map)
        return out
    if expr.is_Pow:
        base, exp = expr.args
        if exp.is_Integer and int(exp) >= 0:
            out = z3.RealVal("1")
            b = _sympy_to_z3(base, var_map)
            for _ in range(int(exp)):
                out = out * b
            return out
    raise ValueError(f"Unsupported expression for z3 conversion: {expr}")


@dataclass
class ToolResult:
    ok: bool
    value: str
    detail: Dict[str, object]


class ToolSandbox:
    def simplify(self, expr: str) -> ToolResult:
        try:
            val = sp.simplify(parse_math_expr(expr))
            return ToolResult(ok=True, value=str(val), detail={})
        except Exception as exc:
            return ToolResult(ok=False, value="", detail={"error": str(exc)})

    def factor(self, expr: str) -> ToolResult:
        try:
            val = sp.factor(parse_math_expr(expr))
            return ToolResult(ok=True, value=str(val), detail={})
        except Exception as exc:
            return ToolResult(ok=False, value="", detail={"error": str(exc)})

    def solve_equation(self, equation: str, variable: Optional[str] = None) -> ToolResult:
        try:
            left, op, right = _split_relation(equation)
            if op not in ("=", "=="):
                raise ValueError("Only equality equations are supported in solve_equation.")
            lhs_expr = parse_math_expr(left)
            rhs_expr = parse_math_expr(right)
            eq = sp.Eq(lhs_expr, rhs_expr)
            if variable is None:
                vars_in_eq = sorted(list(eq.free_symbols), key=lambda x: str(x))
                if not vars_in_eq:
                    return ToolResult(ok=True, value=str(bool(eq)), detail={})
                variable_sym = vars_in_eq[0]
            else:
                variable_sym = sp.Symbol(variable)
            sol = sp.solve(eq, variable_sym)
            return ToolResult(ok=True, value=str(sol), detail={"variable": str(variable_sym)})
        except Exception as exc:
            return ToolResult(ok=False, value="", detail={"error": str(exc)})

    def check_identity(
        self,
        left_expr: str,
        right_expr: str,
        trials: int = 12,
        low: int = -5,
        high: int = 5,
        tol: float = 1e-9,
    ) -> ToolResult:
        try:
            lhs = parse_math_expr(left_expr)
            rhs = parse_math_expr(right_expr)
            symbols = sorted(list(lhs.free_symbols.union(rhs.free_symbols)), key=lambda s: str(s))
            if not symbols:
                ok = sp.simplify(lhs - rhs) == 0
                return ToolResult(ok=bool(ok), value=str(bool(ok)), detail={})
            for _ in range(trials):
                subs = {}
                for sym in symbols:
                    value = 0
                    while value == 0:
                        value = random.randint(low, high)
                    subs[sym] = value
                diff = abs(float(sp.N((lhs - rhs).subs(subs))))
                if diff > tol:
                    return ToolResult(
                        ok=False,
                        value="counterexample",
                        detail={"subs": {str(k): v for k, v in subs.items()}, "diff": diff},
                    )
            return ToolResult(ok=True, value="passed_random_checks", detail={"trials": trials})
        except Exception as exc:
            return ToolResult(ok=False, value="", detail={"error": str(exc)})

    def z3_check_linear_system(
        self,
        constraints: Iterable[str],
    ) -> ToolResult:
        solver = z3.Solver()
        var_map: Dict[str, z3.ArithRef] = {}
        parsed = []
        try:
            for raw in constraints:
                left, op, right = _split_relation(raw)
                lhs_expr = parse_math_expr(left)
                rhs_expr = parse_math_expr(right)
                lhs_z3 = _sympy_to_z3(lhs_expr, var_map)
                rhs_z3 = _sympy_to_z3(rhs_expr, var_map)
                if op in ("=", "=="):
                    constraint = lhs_z3 == rhs_z3
                elif op == ">=":
                    constraint = lhs_z3 >= rhs_z3
                elif op == "<=":
                    constraint = lhs_z3 <= rhs_z3
                elif op == ">":
                    constraint = lhs_z3 > rhs_z3
                elif op == "<":
                    constraint = lhs_z3 < rhs_z3
                elif op == "!=":
                    constraint = lhs_z3 != rhs_z3
                else:
                    raise ValueError(f"Unsupported operator: {op}")
                solver.add(constraint)
                parsed.append(raw)
            sat = solver.check()
            if sat == z3.sat:
                model = solver.model()
                model_data = {str(v): str(model.evaluate(sym, model_completion=True)) for v, sym in var_map.items()}
                return ToolResult(ok=True, value="sat", detail={"model": model_data, "constraints": parsed})
            if sat == z3.unsat:
                return ToolResult(ok=False, value="unsat", detail={"constraints": parsed})
            return ToolResult(ok=False, value="unknown", detail={"constraints": parsed})
        except Exception as exc:
            return ToolResult(ok=False, value="", detail={"error": str(exc), "constraints": parsed})

    def extract_equations(self, text: str) -> List[str]:
        eqs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(op in line for op in ("=", "==", ">=", "<=")) and len(line) <= 120:
                if re.search(r"[A-Za-z0-9\)\]]\s*(==|=|>=|<=)\s*[A-Za-z0-9\(\[]", line):
                    eqs.append(line)
        return eqs
