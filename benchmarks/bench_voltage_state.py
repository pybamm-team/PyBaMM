"""Benchmark voltage observation with default vs legacy voltage-as-state setting.

Demonstrates the observation performance improvement from the new default.

Usage:
    uv run python benchmarks/bench_voltage_state.py
"""

import time

import numpy as np

import pybamm


def bench_observe(model_cls, voltage_as_state: str, n_runs: int = 10):
    """Benchmark voltage observation time."""
    model = model_cls(options={"voltage as a state": voltage_as_state})
    sim = pybamm.Simulation(model)
    t_eval = np.linspace(0, 3600, 10000)

    # Warmup
    sol = sim.solve(t_eval)
    _ = sol["Voltage [V]"].entries

    # Benchmark observation only
    observe_times = []
    for _ in range(n_runs):
        sol = sim.solve(t_eval)
        start = time.perf_counter()
        _ = sol["Voltage [V]"].entries
        observe_times.append(time.perf_counter() - start)

    return np.median(observe_times) * 1000


if __name__ == "__main__":
    models = [
        ("SPM", pybamm.lithium_ion.SPM),
        ("SPMe", pybamm.lithium_ion.SPMe),
        ("DFN", pybamm.lithium_ion.DFN),
    ]

    print("=" * 70)
    print("Voltage Observation Benchmark: Default vs Legacy")
    print("=" * 70)
    print(f"{'Model':<10} {'Default (ms)':<15} {'Legacy (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for model_name, model_cls in models:
        default_ms = bench_observe(model_cls, "true")
        legacy_ms = bench_observe(model_cls, "false")
        speedup = legacy_ms / default_ms
        print(
            f"{model_name:<10} {default_ms:<15.2f} {legacy_ms:<15.2f} {speedup:<10.1f}x"
        )

    print("=" * 70)
    print("Note: Default = voltage as state (new), Legacy = voltage as expression")
    print("=" * 70)
