"""
Benchmarks for the first-call costs a user pays before any result: importing pybamm,
setting up the solver for a new model and reading the first variable.

The rest of the suite times warm, repeated calls, so every round here starts from a
freshly built model.
"""

import os
import subprocess
import sys

import pytest

import pybamm

pytestmark = pytest.mark.time_bench

_MODELS = [
    pytest.param(pybamm.lithium_ion.SPM, id="spm"),
    pytest.param(pybamm.lithium_ion.SPMe, id="spme"),
    pytest.param(pybamm.lithium_ion.DFN, id="dfn"),
]

_T_END = 3000.0


def _built_simulation(model_class):
    sim = pybamm.Simulation(
        model_class(), parameter_values=pybamm.ParameterValues("Chen2020")
    )
    sim.build()
    return sim


def test_import(benchmark):
    # A new interpreter each round; also counts modules, so a stray top-level
    # import of an optional dependency fails its threshold on any hardware
    command = [sys.executable, "-c", "import sys, pybamm; print(len(sys.modules))"]
    # The image skips bytecode; let the warm-up round cache it, as an install would
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}

    def run():
        return subprocess.run(
            command, check=True, capture_output=True, text=True, env=env
        )

    result = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
    module_count = int(result.stdout.split()[-1])
    benchmark.extra_info["measures"] = {"import-modules": module_count}


@pytest.mark.parametrize("model_class", _MODELS)
def test_cold_solve(benchmark, model_class):
    def setup():
        return (_built_simulation(model_class),), {}

    solution = benchmark.pedantic(
        lambda sim: sim.solve([0, _T_END]), setup=setup, rounds=5
    )
    assert solution.termination == "final time"


@pytest.mark.parametrize("model_class", _MODELS)
def test_first_variable_read(benchmark, model_class):
    # Each round reads from a new model, whose observers start empty
    def setup():
        return (_built_simulation(model_class).solve([0, _T_END]),), {}

    def read(solution):
        return solution["Voltage [V]"](solution.t)

    voltage = benchmark.pedantic(read, setup=setup, rounds=5)
    assert voltage.min() > 3.0
