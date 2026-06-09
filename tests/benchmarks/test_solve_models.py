import numpy as np
import numpy.typing as npt
import pytest

import pybamm

# Parameter sets used in the full suite
_ALL_PARAMS = [
    "Marquis2019",
    "ORegan2022",
    "NCA_Kim2011",
    "Prada2013",
    "Ramadass2004",
    "Chen2020",
    "Ecker2015",
]
_PR_PARAMS = ["Marquis2019", "Chen2020"]

_SOLVERS = [pybamm.CasadiSolver, pybamm.IDAKLUSolver]
_SOLVER_IDS = ["casadi", "idaklu"]


def _build_and_discretise(model_class, parameters):
    model = model_class()
    param = pybamm.ParameterValues(parameters)
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
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
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    return model


def _t_eval_and_interp(
    solver: pybamm.BaseSolver, tmax: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    if solver.supports_interp:
        return np.array([0.0, tmax]), None
    return np.linspace(0, tmax, 500), None


# def setup(model_class, parameters, solver, tmax, solve_first):
#     model = _build_and_discretise(model_class, parameters)
#     t_eval, t_interp = _t_eval_and_interp(solver, tmax)
#     if solve_first:
#         solver.solve(model, t_eval=t_eval, t_interp=t_interp)
#     return solver, model, t_eval, t_interp

# ---------------------------------------------------------------------------
# SPM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("parameters", _PR_PARAMS)
@pytest.mark.parametrize("solve_first", [False, True])
def test_solve_spm(benchmark, solve_first, parameters, solver_class):
    solver = solver_class()
    model = _build_and_discretise(pybamm.lithium_ion.SPM, parameters)
    t_eval, t_interp = _t_eval_and_interp(solver, 4000.0)
    if solve_first:
        solver.solve(model, t_eval=t_eval, t_interp=t_interp)
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
@pytest.mark.parametrize("solve_first", [False, True])
def test_solve_spm_full(benchmark, solve_first, parameters, solver_class):
    solver = solver_class()
    model = _build_and_discretise(pybamm.lithium_ion.SPM, parameters)
    t_eval, t_interp = _t_eval_and_interp(solver, 4000.0)
    if solve_first:
        solver.solve(model, t_eval=t_eval, t_interp=t_interp)
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# SPMe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("parameters", _PR_PARAMS)
@pytest.mark.parametrize("solve_first", [False, True])
def test_solve_spme(benchmark, solve_first, parameters, solver_class):
    solver = solver_class()
    model = _build_and_discretise(pybamm.lithium_ion.SPMe, parameters)
    t_eval, t_interp = _t_eval_and_interp(solver, 4000.0)
    if solve_first:
        solver.solve(model, t_eval=t_eval, t_interp=t_interp)
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
@pytest.mark.parametrize("solve_first", [False, True])
def test_solve_spme_full(benchmark, solve_first, parameters, solver_class):
    solver = solver_class()
    model = _build_and_discretise(pybamm.lithium_ion.SPMe, parameters)
    t_eval, t_interp = _t_eval_and_interp(solver, 4000.0)
    if solve_first:
        solver.solve(model, t_eval=t_eval, t_interp=t_interp)
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# DFN
# ---------------------------------------------------------------------------

_DFN_ALL_PARAMS = [
    "Marquis2019",
    "ORegan2022",
    "Prada2013",
    "Ai2020",
    "Ramadass2004",
    "Chen2020",
    "Ecker2015",
]
_DFN_PR_PARAMS = ["Marquis2019", "Chen2020"]


@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("parameters", _DFN_PR_PARAMS)
@pytest.mark.parametrize("solve_first", [False, True])
def test_solve_dfn(benchmark, solve_first, parameters, solver_class):
    if (parameters, solver_class) == ("ORegan2022", pybamm.CasadiSolver):
        pytest.skip("ORegan2022 + CasadiSolver not implemented for DFN")
    solver = solver_class()
    model = _build_and_discretise(pybamm.lithium_ion.DFN, parameters)
    t_eval, t_interp = _t_eval_and_interp(solver, 4000.0)
    if solve_first:
        solver.solve(model, t_eval=t_eval, t_interp=t_interp)
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("parameters", _DFN_ALL_PARAMS)
@pytest.mark.parametrize("solve_first", [False, True])
def test_solve_dfn_full(benchmark, solve_first, parameters, solver_class):
    if (parameters, solver_class) == ("ORegan2022", pybamm.CasadiSolver):
        pytest.skip("ORegan2022 + CasadiSolver not implemented for DFN")
    solver = solver_class()
    model = _build_and_discretise(pybamm.lithium_ion.DFN, parameters)
    t_eval, t_interp = _t_eval_and_interp(solver, 4000.0)
    if solve_first:
        solver.solve(model, t_eval=t_eval, t_interp=t_interp)
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# Repeated solve and voltage observation (IDAKLU only — uses t_interp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    ids=["spm", "spme", "dfn"],
)
@pytest.mark.parametrize("compile_model", [False, True], ids=["no-compile", "compile"])
def test_repeated_solve(benchmark, model_class, compile_model):
    sim = pybamm.Simulation(
        model_class(),
        solver=pybamm.IDAKLUSolver(options={"compile": compile_model}),
    )
    t_eval = [0.0, 3600.0]
    t_interp = np.linspace(t_eval[0], t_eval[-1], 10000)
    sol = sim.solve(t_eval, t_interp=t_interp)
    _ = sol["Voltage [V]"].data  # warm caches

    benchmark(sim.solve, t_eval, t_interp=t_interp)


@pytest.mark.parametrize(
    "model_class",
    [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
    ids=["spm", "spme", "dfn"],
)
@pytest.mark.parametrize("compile_model", [False, True], ids=["no-compile", "compile"])
def test_voltage_observe(benchmark, model_class, compile_model):
    sim = pybamm.Simulation(
        model_class(),
        solver=pybamm.IDAKLUSolver(options={"compile": compile_model}),
    )
    t_eval = [0.0, 3600.0]
    t_interp = np.linspace(t_eval[0], t_eval[-1], 10000)
    sol = sim.solve(t_eval, t_interp=t_interp)
    _ = sol["Voltage [V]"].data  # warm caches

    def run():
        sol._variables.clear()
        return sol["Voltage [V]"].data

    benchmark(run)
