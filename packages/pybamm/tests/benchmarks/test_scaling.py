"""
Benchmarks that time one workload at a small and a large size.

A cost that grows faster than the problem, such as an O(N^2) step loop or a set-up
pass quadratic in the number of states, barely shows at the sizes the rest of the
suite uses. Each pair is also reported as a growth exponent, which unlike the
latencies does not depend on the hardware.
"""

import numpy as np
import pytest

import pybamm

pytestmark = pytest.mark.time_bench


@pytest.fixture(scope="module")
def half_charged_parameters():
    # Balanced cycles from half charge never reach a voltage limit
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values.set_initial_state(0.5)
    return parameter_values


@pytest.mark.parametrize("points", [20, 80])
def test_dfn_mesh_cold_solve(benchmark, record_scaling, points):
    var_pts = {
        "x_n": points,
        "x_s": points,
        "x_p": points,
        "r_n": points,
        "r_p": points,
    }

    def setup():
        sim = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            parameter_values=pybamm.ParameterValues("Chen2020"),
            var_pts=var_pts,
        )
        sim.build()
        return (sim,), {}

    def solve(sim):
        sim.solve([0, 3000])
        return sim

    sim = benchmark.pedantic(solve, setup=setup, rounds=5)
    assert sim.solution.termination == "final time"
    record_scaling(size=sim.built_model.len_rhs_and_alg)


@pytest.mark.parametrize("cycles", [25, 200])
@pytest.mark.parametrize("mode", ["legacy", "unified"])
def test_experiment_cycles(
    benchmark, record_scaling, half_charged_parameters, mode, cycles
):
    experiment = pybamm.Experiment(
        [("Discharge at 1C for 1 minute", "Charge at 1C for 1 minute")] * cycles
    )
    sim = pybamm.Simulation(
        pybamm.lithium_ion.SPM(),
        parameter_values=half_charged_parameters.copy(),
        experiment=experiment,
        experiment_model_mode=mode,
    )
    sim.solve()

    def run():
        solution = sim.solve()
        solution["Voltage [V]"](solution.t)
        return solution

    solution = benchmark(run)
    assert len(solution.cycles) == cycles
    record_scaling(size=2 * cycles, label=mode)


@pytest.mark.parametrize("points", [1_000, 100_000])
def test_output_points(benchmark, record_scaling, points):
    sim = pybamm.Simulation(
        pybamm.lithium_ion.SPM(), parameter_values=pybamm.ParameterValues("Chen2020")
    )
    t_interp = np.linspace(0, 3000, points)
    sim.solve([0, 3000], t_interp=t_interp)

    def run():
        solution = sim.solve([0, 3000], t_interp=t_interp)
        solution["Voltage [V]"](t_interp)
        return solution

    solution = benchmark(run)
    assert len(solution.t) == points
    record_scaling(size=points)
