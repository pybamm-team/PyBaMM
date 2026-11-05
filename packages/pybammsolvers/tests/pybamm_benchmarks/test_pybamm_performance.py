"""Performance benchmarks for PyBaMM integration.

These tests measure the performance of PyBaMM simulations.
They can be run with vanilla PyBaMM or with pybammsolvers installed.
Results are saved to JSON for comparison by the benchmark orchestrator.
"""

import timeit
import json
import os
from pathlib import Path
from datetime import datetime
import pytest

# Check if PyBaMM is available
pytest.importorskip("pybamm", reason="PyBaMM not installed")
import pybamm

# Try to import pybammsolvers (may not be installed for baseline run)
try:
    import pybammsolvers

    PYBAMMSOLVERS_VERSION = pybammsolvers.__version__
except (ImportError, AttributeError):
    PYBAMMSOLVERS_VERSION = None

# Pytest marker for benchmark tests
pytestmark = pytest.mark.benchmark


def time_function(func, num_runs=20):
    """Time a function execution over multiple runs."""
    times = timeit.repeat(func, repeat=5, number=num_runs)
    return {
        "average": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "runs": times,
    }


@pytest.fixture(scope="session")
def performance_results():
    """Session-scoped fixture to collect all performance results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "pybamm_version": pybamm.__version__,
        "pybammsolvers_version": PYBAMMSOLVERS_VERSION,
        "benchmarks": {},
    }
    yield results

    # Save results at end of session
    # Check for environment variable first (set by orchestrator)
    output_file = os.environ.get("BENCHMARK_OUTPUT")
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path("performance_results.json")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Performance results saved to {output_path}")
    print(f"{'=' * 60}")


def record_benchmark(test_name, timing_results, performance_results):
    """Record benchmark results."""
    performance_results["benchmarks"][test_name] = timing_results

    print(f"\n✓ {test_name}:")
    print(f"  Average: {timing_results['average']:.3f}s")
    print(f"  Min: {timing_results['min']:.3f}s, Max: {timing_results['max']:.3f}s")


def test_spm_discharge(performance_results):
    """Benchmark SPM discharge simulation."""

    def run_benchmark():
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())
        sim.solve([0, 3600])

    timing_results = time_function(run_benchmark)
    record_benchmark("SPM 1-hour discharge", timing_results, performance_results)


def test_spm_long_discharge(performance_results):
    """Benchmark longer SPM discharge."""

    def run_benchmark():
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())
        sim.solve([0, 10800])  # 3 hours

    timing_results = time_function(run_benchmark)
    record_benchmark("SPM 3-hour discharge", timing_results, performance_results)


def test_spme_discharge(performance_results):
    """Benchmark SPMe discharge simulation."""

    def run_benchmark():
        model = pybamm.lithium_ion.SPMe()
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())
        sim.solve([0, 3600])

    timing_results = time_function(run_benchmark)
    record_benchmark("SPMe 1-hour discharge", timing_results, performance_results)


def test_dfn_discharge(performance_results):
    """Benchmark DFN discharge simulation."""

    def run_benchmark():
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6))
        sim.solve([0, 1800])

    timing_results = time_function(run_benchmark)
    record_benchmark("DFN 30-min discharge", timing_results, performance_results)


def test_experiment(performance_results):
    """Benchmark experiment simulation."""

    def run_benchmark():
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C for 30 minutes",
                "Rest for 10 minutes",
                "Charge at 0.5C for 30 minutes",
            ]
        )
        sim = pybamm.Simulation(
            model, experiment=experiment, solver=pybamm.IDAKLUSolver()
        )
        sim.solve()

    timing_results = time_function(run_benchmark)
    record_benchmark("Simple experiment", timing_results, performance_results)


def test_multiple_solves(performance_results):
    """Benchmark multiple sequential solves."""

    def run_benchmark():
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.IDAKLUSolver()

        for _ in range(10):
            sim = pybamm.Simulation(model, solver=solver)
            sim.solve([0, 1800])

    timing_results = time_function(run_benchmark)
    record_benchmark("10 sequential SPM solves", timing_results, performance_results)
