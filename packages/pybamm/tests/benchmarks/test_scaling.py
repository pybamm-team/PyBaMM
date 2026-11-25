"""Workloads at a small and a large size, to catch super-linear growth."""

import numpy as np
import pytest

import pybamm

pytestmark = pytest.mark.time_bench


@pytest.fixture(scope="module")
def half_charged_parameters():
    # Keeps the balanced cycles clear of the voltage limits
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


def _pointwise_model(points, algebraic):
    """A linear ODE, or DAE, uncoupled across ``points`` cells, so each costs the same."""
    model = pybamm.BaseModel()
    x = pybamm.SpatialVariable("x", domain="negative electrode")
    y = pybamm.Variable("y", domain="negative electrode")
    rate = pybamm.InputParameter("rate")
    if algebraic:
        z = pybamm.Variable("z", domain="negative electrode")
        model.rhs = {y: -rate * y + 0.1 * z}
        model.algebraic = {z: z - y}
        model.initial_conditions = {y: 1, z: 1}
    else:
        model.rhs = {y: -rate * y}
        model.initial_conditions = {y: 1}
    model.variables = {"y": y}
    geometry = {"negative electrode": {x: {"min": 0, "max": 1}}}
    submesh_types = {"negative electrode": pybamm.Uniform1DSubMesh}
    mesh = pybamm.Mesh(geometry, submesh_types, {x: points})
    spatial_methods = {"negative electrode": pybamm.FiniteVolume()}
    pybamm.Discretisation(mesh, spatial_methods).process_model(model)
    return model


@pytest.mark.parametrize("points", [2_000, 32_000])
@pytest.mark.parametrize("kind", ["ode", "dae"])
def test_first_sensitivity_solve(
    benchmark, record_scaling, record_peak_memory, kind, points
):
    # A cost super-linear in the state count hides behind the linear ones at the
    # battery models' default sizes; 32,000 cells expose it
    def setup():
        model = _pointwise_model(points, algebraic=kind == "dae")
        return (model, pybamm.IDAKLUSolver()), {}

    def solve(model, solver):
        return solver.solve(
            model, [0, 1], inputs={"rate": 1.0}, calculate_sensitivities=True
        )

    solution = benchmark.pedantic(solve, setup=setup, rounds=5)
    assert solution.termination == "final time"
    assert "rate" in solution.sensitivities
    record_scaling(size=points, label=kind)
    record_peak_memory(solve, setup)


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
    assert [len(cycle.steps) for cycle in solution.cycles] == experiment.cycle_lengths
    assert solution.termination == "final time"
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
