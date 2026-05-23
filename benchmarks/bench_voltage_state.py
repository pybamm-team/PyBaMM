"""Benchmark voltage state lookup vs expression observation.

Compares the cost of reading Voltage [V] (a state, O(1) lookup) against
Voltage expression [V] (requires post-solve computation).

Usage:
    uv run python benchmarks/bench_voltage_state.py
"""

import statistics
import time

import numpy as np

import pybamm

N_PTS = 10_000
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

    t_interp = np.linspace(0, 3600, N_PTS)

    print("=" * 70)
    print("Voltage State Lookup vs Expression Observation")
    print(f"  t_interp={N_PTS} points, {N_RUNS} runs, solver=IDAKLUSolver")
    print("=" * 70)
    print(f"{'Model':<6} {'V state [ms]':>13} {'V expr [ms]':>13} {'Speedup':>8}")
    print("-" * 45)

    for name, cls in models:
        sim = pybamm.Simulation(cls(), solver=pybamm.IDAKLUSolver())
        sol = sim.solve(t_eval=[0, 3600], t_interp=t_interp)
        state_ms = bench_observe(sol, "Voltage [V]") * 1000
        expr_ms = bench_observe(sol, "Voltage expression [V]") * 1000
        speedup = expr_ms / state_ms if state_ms > 0 else float("inf")
        print(f"{name:<6} {state_ms:>13.3f} {expr_ms:>13.3f} {speedup:>7.1f}x")

    print("=" * 70)
