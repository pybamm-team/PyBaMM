import pytest

import pybamm

_PARAMS = [
    "Marquis2019",
    pytest.param("ORegan2022", marks=pytest.mark.slow_bench),
    pytest.param("NCA_Kim2011", marks=pytest.mark.slow_bench),  # Not DFN
    pytest.param("Prada2013", marks=pytest.mark.slow_bench),
    pytest.param("Ai2020", marks=pytest.mark.slow_bench),  # DFN only
    pytest.param("Ramadass2004", marks=pytest.mark.slow_bench),
    pytest.param("Mohtat2020", marks=pytest.mark.slow_bench),
    "Chen2020",
    pytest.param("OKane2022", marks=pytest.mark.slow_bench),
    pytest.param("Ecker2015", marks=pytest.mark.slow_bench),
]


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


@pytest.mark.parametrize("parameters", _PARAMS)
def test_setup_spm(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.SPM, parameters)


@pytest.mark.parametrize("parameters", _PARAMS)
def test_setup_spme(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.SPMe, parameters)


@pytest.mark.parametrize("parameters", _PARAMS)
def test_setup_dfn(benchmark, parameters):
    benchmark(_build, pybamm.lithium_ion.DFN, parameters)


@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _PARAMS)
def test_setup_spm_simulation(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.SPM, parameters, with_experiment)


@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _PARAMS)
def test_setup_spme_simulation(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.SPMe, parameters, with_experiment)


@pytest.mark.parametrize("with_experiment", [False, True], ids=["no-exp", "with-exp"])
@pytest.mark.parametrize("parameters", _PARAMS)
def test_setup_dfn_simulation(benchmark, with_experiment, parameters):
    benchmark(_build_sim, pybamm.lithium_ion.DFN, parameters, with_experiment)
