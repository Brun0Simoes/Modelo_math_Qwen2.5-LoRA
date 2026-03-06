from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(r"E:\IA_matematica")
MODULE_PATH = PROJECT_ROOT / "scripts" / "solve_problem_json.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("solve_problem_json", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> None:
    mod = _load_module()

    median_problem = """Uma empresa de tecnologia vai padronizar a velocidade de
conexao de internet que oferece a seus clientes em dez cidades.
A direcao da empresa decide que seu novo padrao de velocidade de
referencia sera a mediana dos valores das velocidades de referencia
de conexoes nessas dez cidades. Esses valores, em megabyte
por segundo (MB/s), sao apresentados no quadro.
Cidades Velocidade de referencia (MB/s)
C1 390
C2 380
C3 320
C4 390
C5 340
C6 380
C7 390
C8 400
C9 350
C10 360
A velocidade de referencia, em megabyte por segundo, a ser
adotada por essa empresa e
A 360
B 370
C 380
D 390
E 400"""
    median_opts = mod._extract_options(median_problem)
    median_inf = mod._infer_statistical_objective(median_problem)
    _assert(median_inf is not None, "Median inference failed")
    _assert(median_inf["target"] == "median", "Median target mismatch")
    _assert(abs(float(median_inf["result"]) - 380.0) < 1e-9, "Median result mismatch")
    median_cand = mod._make_fallback_candidate_statistics(median_problem, median_inf, median_opts)
    _assert("380" in median_cand["final_answer"], "Median fallback final answer mismatch")
    _assert("option C" in median_cand["final_answer"], "Median fallback option mismatch")

    mean_problem = """As velocidades medidas em cinco cidades foram:
10, 20, 30, 40, 50.
A media aritmetica dessas velocidades e:
A 20
B 25
C 30
D 35
E 40"""
    mean_opts = mod._extract_options(mean_problem)
    mean_inf = mod._infer_statistical_objective(mean_problem)
    _assert(mean_inf is not None, "Mean inference failed")
    _assert(mean_inf["target"] == "mean", "Mean target mismatch")
    _assert(abs(float(mean_inf["result"]) - 30.0) < 1e-9, "Mean result mismatch")
    mean_cand = mod._make_fallback_candidate_statistics(mean_problem, mean_inf, mean_opts)
    _assert("30" in mean_cand["final_answer"], "Mean fallback final answer mismatch")
    _assert("option C" in mean_cand["final_answer"], "Mean fallback option mismatch")

    mode_problem = """Valores observados: 1, 1, 2, 3, 3, 3, 4.
A moda e:
A 1
B 2
C 3
D 4
E 5"""
    mode_opts = mod._extract_options(mode_problem)
    mode_inf = mod._infer_statistical_objective(mode_problem)
    _assert(mode_inf is not None, "Mode inference failed")
    _assert(mode_inf["target"] == "mode", "Mode target mismatch")
    _assert(abs(float(mode_inf["result"]) - 3.0) < 1e-9, "Mode result mismatch")
    mode_cand = mod._make_fallback_candidate_statistics(mode_problem, mode_inf, mode_opts)
    _assert("3" in mode_cand["final_answer"], "Mode fallback final answer mismatch")
    _assert("option C" in mode_cand["final_answer"], "Mode fallback option mismatch")

    range_problem = """Dados de temperatura:
11, 15, 19, 14, 20.
A amplitude (range) dos dados e:
A 7
B 8
C 9
D 10
E 11"""
    range_opts = mod._extract_options(range_problem)
    range_inf = mod._infer_statistical_objective(range_problem)
    _assert(range_inf is not None, "Range inference failed")
    _assert(range_inf["target"] == "range", "Range target mismatch")
    _assert(abs(float(range_inf["result"]) - 9.0) < 1e-9, "Range result mismatch")
    range_cand = mod._make_fallback_candidate_statistics(range_problem, range_inf, range_opts)
    _assert("9" in range_cand["final_answer"], "Range fallback final answer mismatch")
    _assert("option C" in range_cand["final_answer"], "Range fallback option mismatch")

    no_stats_problem = """Sejam a, b, c > 0. Prove que
a/sqrt(a^2+8bc) + b/sqrt(b^2+8ca) + c/sqrt(c^2+8ab) >= 1."""
    no_stats_inf = mod._infer_statistical_objective(no_stats_problem)
    _assert(no_stats_inf is None, "False positive in statistics inference")

    loss_problem = """Uma distribuidora encomendou 10000 litros.
Enviou 10200 litros e entregou 9900 litros.
No novo pedido, solicitou o dobro do volume encomendado no pedido anterior.
Qual o volume minimo a enviar?
A 20100
B 20200
C 20300
D 20400
E 20600"""
    loss_inf = mod._infer_constant_loss_transport(loss_problem)
    _assert(loss_inf is not None, "Constant loss inference failed")
    _assert(abs(float(loss_inf["gross_to_send"]) - 20300.0) < 1e-9, "Constant loss result mismatch")

    print("All fallback tests passed.")


if __name__ == "__main__":
    run()
