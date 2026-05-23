"""Benchmark voltage observation and experiment event alignment by surface form.

Focuses on the reduced-order lithium-ion models where voltage-as-a-state changes
event handling the most: SPM and SPMe.

For each model and surface form, the script reports:
1. Post-solve observe cost for ``Voltage [V]`` versus ``Voltage expression [V]``
2. Legacy/unified event timing drift on the CC/CV experiment that currently fails

Usage:
    .venv/bin/python benchmarks/bench_voltage_state_surface_forms.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import pybamm

SURFACE_FORMS = ("false", "differential", "algebraic")
MODELS = (
    ("SPM", pybamm.lithium_ion.SPM),
    ("SPMe", pybamm.lithium_ion.SPMe),
)
N_PTS = 10_000
OBSERVE_RUNS = 15
INITIAL_SOC = 0.2
EXPERIMENT = pybamm.Experiment(
    [("Charge at C/3 until 4.1 V", "Hold at 4.1 V until C/20")]
)
STEP_RTOL = 5e-5
STEP_ATOL = 5e-4
TOTAL_RTOL = 5e-5
TOTAL_ATOL = 5e-4


def clear_observation_cache(sol, variable):
    """Evict one processed variable so observe timing measures fresh work."""
    sol._variables.pop(variable, None)


def bench_observe(sol, variable, n_runs=OBSERVE_RUNS):
    """Return median observe time in seconds for a single solved variable."""
    _ = sol[variable].entries

    times = []
    for _ in range(n_runs):
        clear_observation_cache(sol, variable)
        start = time.perf_counter()
        _ = sol[variable].entries
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def make_model(model_cls, surface_form):
    return model_cls({"surface form": surface_form})


def observe_rows():
    t_interp = np.linspace(0, 3600, N_PTS)

    for model_name, model_cls in MODELS:
        for surface_form in SURFACE_FORMS:
            sim = pybamm.Simulation(
                make_model(model_cls, surface_form),
                solver=pybamm.IDAKLUSolver(),
            )
            start = time.perf_counter()
            sol = sim.solve(t_eval=[0, 3600], t_interp=t_interp)
            solve_s = time.perf_counter() - start
            v_state_ms = bench_observe(sol, "Voltage [V]") * 1000
            v_expr_ms = bench_observe(sol, "Voltage expression [V]") * 1000
            yield {
                "model": model_name,
                "surface_form": surface_form,
                "solve_s": solve_s,
                "v_state_ms": v_state_ms,
                "v_expr_ms": v_expr_ms,
                "speedup": v_expr_ms / v_state_ms if v_state_ms > 0 else float("inf"),
            }


def within_tol(actual, expected, rtol, atol):
    return abs(actual - expected) <= atol + rtol * abs(expected)


def event_rows():
    for model_name, model_cls in MODELS:
        for surface_form in SURFACE_FORMS:
            results = {}
            for mode in ("legacy", "unified"):
                sim = pybamm.Simulation(
                    make_model(model_cls, surface_form),
                    experiment=EXPERIMENT,
                    solver=pybamm.IDAKLUSolver(),
                    experiment_model_mode=mode,
                )
                sol = sim.solve(calc_esoh=False, initial_soc=INITIAL_SOC)
                step1 = sol.cycles[0].steps[0]
                step2 = sol.cycles[0].steps[1]
                results[mode] = {
                    "step_ts": [float(step1.t[-1]), float(step2.t[-1])],
                    "total_t": float(sol.t[-1]),
                    "terminations": [step1.termination, step2.termination],
                }

            step1_dt_s = (
                results["unified"]["step_ts"][0] - results["legacy"]["step_ts"][0]
            )
            step2_dt_s = (
                results["unified"]["step_ts"][1] - results["legacy"]["step_ts"][1]
            )
            total_dt_s = results["unified"]["total_t"] - results["legacy"]["total_t"]
            step_passes = [
                within_tol(
                    legacy_t,
                    unified_t,
                    STEP_RTOL,
                    STEP_ATOL,
                )
                for legacy_t, unified_t in zip(
                    results["legacy"]["step_ts"],
                    results["unified"]["step_ts"],
                    strict=True,
                )
            ]
            yield {
                "model": model_name,
                "surface_form": surface_form,
                "step1_dt_s": step1_dt_s,
                "step1_pass": step_passes[0],
                "step2_dt_s": step2_dt_s,
                "step2_pass": step_passes[1],
                "total_dt_s": total_dt_s,
                "total_pass": within_tol(
                    results["legacy"]["total_t"],
                    results["unified"]["total_t"],
                    TOTAL_RTOL,
                    TOTAL_ATOL,
                ),
                "termination_match": (
                    results["legacy"]["terminations"]
                    == results["unified"]["terminations"]
                ),
                "test_pass": (
                    step_passes[0]
                    and step_passes[1]
                    and within_tol(
                        results["legacy"]["total_t"],
                        results["unified"]["total_t"],
                        TOTAL_RTOL,
                        TOTAL_ATOL,
                    )
                    and (
                        results["legacy"]["terminations"]
                        == results["unified"]["terminations"]
                    )
                ),
            }


def root_method_rows():
    t_interp = np.linspace(0, 3600, N_PTS)

    for model_name, model_cls in MODELS:
        for rm_label, rm_value in [
            ("nonlinear_solver", "nonlinear_solver"),
            ("None", None),
        ]:
            sim = pybamm.Simulation(
                model_cls(),
                solver=pybamm.IDAKLUSolver(root_method=rm_value),
            )
            times = []
            for _ in range(5):
                start = time.perf_counter()
                sim.solve(t_eval=[0, 3600], t_interp=t_interp)
                times.append(time.perf_counter() - start)
            yield {
                "model": model_name,
                "root_method": rm_label,
                "median_solve_ms": statistics.median(times) * 1000,
            }


def print_root_method_table(rows):
    print()
    print("Root method comparison (IDAKLUSolver)")
    print(f"{'Model':<6} {'root_method':<20} {'Median solve [ms]':>18}")
    print("-" * 48)
    for row in rows:
        print(
            f"{row['model']:<6} {row['root_method']:<20} "
            f"{row['median_solve_ms']:>18.1f}"
        )


def print_observe_table(rows):
    print(
        "Observe benchmark"
        f"  (t_interp={N_PTS}, runs={OBSERVE_RUNS}, solver=IDAKLUSolver)"
    )
    print(
        f"{'Model':<5} {'Surface form':<13} {'Solve [ms]':>10} "
        f"{'V state [ms]':>13} {'V expr [ms]':>12} {'Speedup':>8}"
    )
    print("-" * 70)
    for row in rows:
        print(
            f"{row['model']:<5} {row['surface_form']:<13} {row['solve_s'] * 1000:>10.3f} "
            f"{row['v_state_ms']:>13.3f} {row['v_expr_ms']:>12.3f} {row['speedup']:>7.1f}x"
        )


def print_event_table(rows):
    print()
    print("Event alignment benchmark")
    print("  Experiment: Charge at C/3 until 4.1 V, then Hold at 4.1 V until C/20")
    print(
        f"  Pass criteria match {STEP_RTOL:g} rtol / {STEP_ATOL:g} atol "
        "from test_run_event_driven_experiment_unified_matches_legacy"
    )
    print(
        f"{'Model':<5} {'Surface form':<13} {'Step1 dt [s]':>13} {'Step1 pass':>11} "
        f"{'Step2 dt [s]':>13} {'Step2 pass':>11} {'Total dt [s]':>13} "
        f"{'Test pass':>10}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{row['model']:<5} {row['surface_form']:<13} {row['step1_dt_s']:>13.6f} "
            f"{row['step1_pass']!s:>11} {row['step2_dt_s']:>13.6f} "
            f"{row['step2_pass']!s:>11} {row['total_dt_s']:>13.6f} "
            f"{row['test_pass']!s:>10}"
        )


if __name__ == "__main__":
    observe = list(observe_rows())
    events = list(event_rows())
    print_observe_table(observe)
    print_event_table(events)
    root_methods = list(root_method_rows())
    print_root_method_table(root_methods)
