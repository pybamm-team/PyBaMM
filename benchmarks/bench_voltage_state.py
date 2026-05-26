"""Benchmark voltage state lookup vs expression observation.

Compares the cost of reading Voltage [V] (a state, O(1) lookup) against
Voltage expression [V] (requires post-solve computation) as a function
of the number of t_eval points.

Usage:
    uv run python benchmarks/bench_voltage_state.py
"""

import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pybamm

T_EVAL_SIZES = [100, 500, 1000, 5000, 10_000]
N_RUNS = 15


def clear_cache(sol, variable):
    sol._variables.pop(variable, None)


def bench_observe(sol, variable, n_runs=N_RUNS):
    _ = sol[variable].entries  # warm up

    times = []
    for _ in range(n_runs):
        clear_cache(sol, variable)
        start = time.perf_counter()
        _ = sol[variable].entries
        times.append(time.perf_counter() - start)
    return statistics.median(times)


if __name__ == "__main__":
    models = [
        ("SPM", pybamm.lithium_ion.SPM),
        ("SPMe", pybamm.lithium_ion.SPMe),
        ("DFN", pybamm.lithium_ion.DFN),
    ]

    results = {}
    for name, cls in models:
        state_times = []
        expr_times = []
        for n_pts in T_EVAL_SIZES:
            t_interp = np.linspace(0, 3600, n_pts)
            sim = pybamm.Simulation(cls(), solver=pybamm.IDAKLUSolver())
            sol = sim.solve(t_eval=[0, 3600], t_interp=t_interp)

            state_ms = bench_observe(sol, "Voltage [V]") * 1000
            expr_ms = bench_observe(sol, "Voltage expression [V]") * 1000
            state_times.append(state_ms)
            expr_times.append(expr_ms)
            print(f"{name} n={n_pts:>5d}  state={state_ms:.3f}ms  expr={expr_ms:.3f}ms")

        results[name] = (state_times, expr_times)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, (state_times, expr_times)) in zip(
        axes, results.items(), strict=False
    ):
        ax.plot(T_EVAL_SIZES, state_times, "o-", label="State lookup")
        ax.plot(T_EVAL_SIZES, expr_times, "o-", label="Expression eval")
        ax.set_title(name)
        ax.set_xlabel("t_eval points")
        ax.set_ylabel("Observe time [ms]")
        ax.legend()

    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")
