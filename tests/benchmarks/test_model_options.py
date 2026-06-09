"""
Benchmarks for different model option combinations.

All tests here are marked `slow` — the full option sweep is large and
intended for the main-branch history run, not PR CI.
"""

import numpy as np
import numpy.typing as npt
import pytest

import pybamm

_MODELS = [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN]
_MODEL_IDS = ["spm", "dfn"]
_SOLVERS = [pybamm.CasadiSolver, pybamm.IDAKLUSolver]
_SOLVER_IDS = ["casadi", "idaklu"]


def _build_model(parameter, model_class, option, value, additional_params=None):
    param = pybamm.ParameterValues(parameter)
    if additional_params:
        param.update(additional_params)
    model = model_class({option: value})
    param.process_model(model)
    var_pts = {
        "x_n": 20,
        "x_s": 20,
        "x_p": 20,
        "r_n": 30,
        "r_p": 30,
        "y": 10,
        "z": 10,
    }
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    return model


def _solve_setup(
    parameter,
    model_class,
    option,
    value,
    solver_class,
    additional_params=None,
) -> tuple[
    pybamm.BaseSolver,
    pybamm.BaseModel,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
]:
    solver = solver_class()
    tmax = 4000.0
    if solver.supports_interp:
        t_eval = np.array([0.0, tmax])
        t_interp = None
    else:
        t_eval = np.linspace(0, tmax, 500)
        t_interp = None
    model = _build_model(parameter, model_class, option, value, additional_params)
    return solver, model, t_eval, t_interp


# ---------------------------------------------------------------------------
# Loss of active material
# ---------------------------------------------------------------------------

_LAM_OPTIONS = [
    "none",
    "stress-driven",
    "reaction-driven",
    "stress and reaction-driven",
]


@pytest.mark.slow_bench
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _LAM_OPTIONS)
def test_build_loss_active_material(benchmark, model_class, option):
    benchmark(_build_model, "Ai2020", model_class, "loss of active material", option)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _LAM_OPTIONS)
def test_solve_loss_active_material(benchmark, model_class, option, solver_class):
    if (model_class, solver_class) == (pybamm.lithium_ion.DFN, pybamm.CasadiSolver):
        pytest.skip("DFN + CasadiSolver is too slow!")
    solver, model, t_eval, t_interp = _solve_setup(
        "Ai2020", model_class, "loss of active material", option, solver_class
    )
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# Lithium plating
# ---------------------------------------------------------------------------

_PLATING_OPTIONS = ["none", "irreversible", "reversible", "partially reversible"]


@pytest.mark.slow_bench
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _PLATING_OPTIONS)
def test_build_lithium_plating(benchmark, model_class, option):
    benchmark(_build_model, "OKane2022", model_class, "lithium plating", option)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _PLATING_OPTIONS)
def test_solve_lithium_plating(benchmark, model_class, option, solver_class):
    solver, model, t_eval, t_interp = _solve_setup(
        "OKane2022", model_class, "lithium plating", option, solver_class
    )
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# SEI
# ---------------------------------------------------------------------------

_SEI_OPTIONS = [
    "none",
    "constant",
    "reaction limited",
    "solvent-diffusion limited",
    "electron-migration limited",
    "interstitial-diffusion limited",
    "ec reaction limited",
]
_SEI_TUNNELLING_OPTIONS = ["tunnelling limited", "VonKolzenberg2020"]
_SEI_TUNNELLING_EXTRA = {
    "tunnelling limited": {"Tunneling barrier factor [m-1]": 6.0e9},
    "VonKolzenberg2020": {
        "Tunneling distance for electrons [m]": 0,
        "SEI lithium ion conductivity [S.m-1]": 1.0e-7,
    },
}


@pytest.mark.slow_bench
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _SEI_OPTIONS)
def test_build_sei(benchmark, model_class, option):
    benchmark(_build_model, "Marquis2019", model_class, "SEI", option)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _SEI_OPTIONS)
def test_solve_sei(benchmark, model_class, option, solver_class):
    solver, model, t_eval, t_interp = _solve_setup(
        "Marquis2019", model_class, "SEI", option, solver_class
    )
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _SEI_TUNNELLING_OPTIONS)
def test_solve_sei_tunnelling(benchmark, model_class, option, solver_class):
    solver, model, t_eval, t_interp = _solve_setup(
        "Chen2020",
        model_class,
        "SEI",
        option,
        solver_class,
        additional_params=_SEI_TUNNELLING_EXTRA[option],
    )
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# Particle diffusion
# ---------------------------------------------------------------------------

_PARTICLE_OPTIONS = [
    "Fickian diffusion",
    "uniform profile",
    "quadratic profile",
    "quartic profile",
]


@pytest.mark.slow_bench
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _PARTICLE_OPTIONS)
def test_build_particle(benchmark, model_class, option):
    benchmark(_build_model, "Marquis2019", model_class, "particle", option)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _PARTICLE_OPTIONS)
def test_solve_particle(benchmark, model_class, option, solver_class):
    solver, model, t_eval, t_interp = _solve_setup(
        "Marquis2019", model_class, "particle", option, solver_class
    )
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------

_THERMAL_OPTIONS = ["isothermal", "lumped", "x-full"]


@pytest.mark.slow_bench
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _THERMAL_OPTIONS)
def test_build_thermal(benchmark, model_class, option):
    benchmark(_build_model, "Marquis2019", model_class, "thermal", option)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _THERMAL_OPTIONS)
def test_solve_thermal(benchmark, model_class, option, solver_class):
    solver, model, t_eval, t_interp = _solve_setup(
        "Marquis2019", model_class, "thermal", option, solver_class
    )
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)


# ---------------------------------------------------------------------------
# Surface form
# ---------------------------------------------------------------------------

_SURFACE_FORM_OPTIONS = ["false", "differential", "algebraic"]


@pytest.mark.slow_bench
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _SURFACE_FORM_OPTIONS)
def test_build_surface_form(benchmark, model_class, option):
    benchmark(_build_model, "Marquis2019", model_class, "surface form", option)


@pytest.mark.slow_bench
@pytest.mark.parametrize("solver_class", _SOLVERS, ids=_SOLVER_IDS)
@pytest.mark.parametrize("model_class", _MODELS, ids=_MODEL_IDS)
@pytest.mark.parametrize("option", _SURFACE_FORM_OPTIONS)
def test_solve_surface_form(benchmark, model_class, option, solver_class):
    if (model_class, option, solver_class) == (
        pybamm.lithium_ion.SPM,
        "differential",
        pybamm.IDAKLUSolver,
    ):
        pytest.skip("SPM + differential + IDAKLUSolver not implemented")
    solver, model, t_eval, t_interp = _solve_setup(
        "Marquis2019", model_class, "surface form", option, solver_class
    )
    benchmark(solver.solve, model, t_eval=t_eval, t_interp=t_interp)
