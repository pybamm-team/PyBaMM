import pytest

import pybamm

_ALL_PARAMS = [
    "Marquis2019",
    "ORegan2022",
    "NCA_Kim2011",
    "Prada2013",
    "Ai2020",
    "Ramadass2004",
    "Mohtat2020",
    "Chen2020",
    "OKane2022",
    "Ecker2015",
]
_PR_PARAMS = ["Marquis2019", "Chen2020"]


def _compute_discretisation(model, param):
    var_pts = {
        pybamm.standard_spatial_vars.x_n: 20,
        pybamm.standard_spatial_vars.x_s: 20,
        pybamm.standard_spatial_vars.x_p: 20,
        pybamm.standard_spatial_vars.r_n: 30,
        pybamm.standard_spatial_vars.r_p: 30,
        pybamm.standard_spatial_vars.y: 10,
        pybamm.standard_spatial_vars.z: 10,
    }
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    return pybamm.Discretisation(mesh, model.default_spatial_methods)


def _build(model_class, parameters):
    param = pybamm.ParameterValues(parameters)
    model = model_class()
    param.process_model(model)
    _compute_discretisation(model, param).process_model(model)


def _build_sim(model_class, parameters, with_experiment):
    param = pybamm.ParameterValues(parameters)
    model = model_class()
    if with_experiment:
        exp = pybamm.Experiment(["Discharge at 0.1C until 3.105 V"])
        pybamm.Simulation(model, parameter_values=param, experiment=exp)
    else:
        pybamm.Simulation(model, parameter_values=param, C_rate=1)


# ---------------------------------------------------------------------------
# Model build benchmarks (PR subset)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parameters", _PR_PARAMS)
def test_setup_spm(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.SPM, parameters)


@pytest.mark.parametrize("parameters", _PR_PARAMS)
def test_setup_spme(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.SPMe, parameters)


@pytest.mark.parametrize("parameters", _PR_PARAMS)
def test_setup_dfn(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.DFN, parameters)


# ---------------------------------------------------------------------------
# Model build benchmarks (full grid)
# ---------------------------------------------------------------------------


@pytest.mark.slow_bench
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
def test_setup_spm_full(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.SPM, parameters)


@pytest.mark.slow_bench
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
def test_setup_spme_full(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.SPMe, parameters)


@pytest.mark.slow_bench
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
def test_setup_dfn_full(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.DFN, parameters)


# ---------------------------------------------------------------------------
# Simulation setup benchmarks (PR subset)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _PR_PARAMS)
def test_setup_spm_simulation(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.SPM, parameters, with_experiment)


@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _PR_PARAMS)
def test_setup_spme_simulation(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.SPMe, parameters, with_experiment)


@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _PR_PARAMS)
def test_setup_dfn_simulation(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.DFN, parameters, with_experiment)


# ---------------------------------------------------------------------------
# Simulation setup benchmarks (full grid)
# ---------------------------------------------------------------------------


@pytest.mark.slow_bench
@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
def test_setup_spm_simulation_full(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.SPM, parameters, with_experiment)


@pytest.mark.slow_bench
@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
def test_setup_spme_simulation_full(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.SPMe, parameters, with_experiment)


@pytest.mark.slow_bench
@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _ALL_PARAMS)
def test_setup_dfn_simulation_full(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.DFN, parameters, with_experiment)
