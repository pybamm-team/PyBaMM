import logging

import numpy as np
import pytest

import pybamm

pytestmark = pytest.mark.time_bench

# Per push: Marquis2019, Chen2020 and the stiffest set, ORegan2022. The rest
# run on the scheduled sweep.
_PARAMS = [
    "Marquis2019",
    "Chen2020",
    "ORegan2022",
    *(
        pytest.param(parameters, marks=pytest.mark.slow_bench)
        for parameters in [
            "NCA_Kim2011",
            "Prada2013",
            "Ai2020",
            "Ramadass2004",
            "Ecker2015",
        ]
    ),
]

_T_MAX = 4000.0


def _build_and_discretise(model_class, parameters):
    model = model_class()
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(parameters)
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {
        "x_n": 20,
        "x_s": 20,
        "x_p": 20,
        "r_n": 30,
        "r_p": 30,
        "y": 10,
        "z": 10,
    }
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    return model


def _assert_discharged(solution):
    # A solve that stops early would otherwise time as a speed-up
    assert solution.termination in ("final time", "event: Minimum voltage [V]")
    assert solution.t[-1] > 0.5 * _T_MAX


def _assert_compiled(caplog):
    # Without a C compiler the solver silently falls back to the uncompiled path
    failures = [r for r in caplog.records if "Failed to compile" in r.getMessage()]
    assert not failures, failures[0].getMessage()


def _benchmark_warm_solve(benchmark, model_class, parameters):
    solver = pybamm.IDAKLUSolver()
    model = _build_and_discretise(model_class, parameters)
    t_eval = np.array([0.0, _T_MAX])
    # Set up outside the timing; test_cold_start.py times the set-up
    solver.solve(model, t_eval=t_eval)
    solution = benchmark(solver.solve, model, t_eval=t_eval)
    _assert_discharged(solution)


@pytest.mark.parametrize("parameters", _PARAMS)
def test_solve_spm(benchmark, parameters):
    _benchmark_warm_solve(benchmark, pybamm.lithium_ion.SPM, parameters)


@pytest.mark.parametrize("parameters", _PARAMS)
def test_solve_spme(benchmark, parameters):
    _benchmark_warm_solve(benchmark, pybamm.lithium_ion.SPMe, parameters)


@pytest.mark.parametrize("parameters", _PARAMS)
def test_solve_dfn(benchmark, parameters):
    _benchmark_warm_solve(benchmark, pybamm.lithium_ion.DFN, parameters)


# ---------------------------------------------------------------------------
# Repeated solve and voltage observation (IDAKLU only — uses t_interp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    ids=["spm", "spme", "dfn"],
)
@pytest.mark.parametrize("compile_model", [False, True], ids=["no-compile", "compile"])
def test_repeated_solve(benchmark, caplog, model_class, compile_model):
    sim = pybamm.Simulation(
        model_class(),
        solver=pybamm.IDAKLUSolver(options={"compile": compile_model}),
    )
    t_eval = [0.0, 3600.0]
    t_interp = np.linspace(t_eval[0], t_eval[-1], 10000)
    # Warm the casadi/AOT caches and the voltage observer before timing.
    with caplog.at_level(logging.WARNING):
        sol = sim.solve(t_eval, t_interp=t_interp)
        _ = sol["Voltage [V]"].data  # warm caches
    if compile_model:
        _assert_compiled(caplog)

    benchmark(sim.solve, t_eval, t_interp=t_interp)


@pytest.mark.parametrize(
    "model_class",
    [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    ids=["spm", "spme", "dfn"],
)
@pytest.mark.parametrize("compile_model", [False, True], ids=["no-compile", "compile"])
def test_voltage_observe(benchmark, caplog, model_class, compile_model):
    sim = pybamm.Simulation(
        model_class(),
        solver=pybamm.IDAKLUSolver(options={"compile": compile_model}),
    )
    t_eval = [0.0, 3600.0]
    t_interp = np.linspace(t_eval[0], t_eval[-1], 10000)
    # Warm the casadi/AOT caches and the voltage observer before timing.
    with caplog.at_level(logging.WARNING):
        sol = sim.solve(t_eval, t_interp=t_interp)
        _ = sol["Voltage [V]"].data  # warm caches
    if compile_model:
        _assert_compiled(caplog)

    def run():
        sol._variables.clear()
        return sol["Voltage [V]"].data

    benchmark(run)
