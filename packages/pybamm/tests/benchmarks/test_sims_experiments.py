import pytest

import pybamm

pytestmark = pytest.mark.speed_bench

_EXPERIMENT_DESCRIPTIONS = {
    "CCCV": [
        "Discharge at C/5 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 10 mA",
        "Rest for 1 hour",
    ],
    "GITT": [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 10,
}

_MODELS = [
    pytest.param(pybamm.lithium_ion.SPM, id="spm"),
    pytest.param(pybamm.lithium_ion.DFN, id="dfn"),
]


def _setup_sim(experiment, parameters, model_class):
    param = pybamm.ParameterValues(parameters)
    model = model_class()
    solver = pybamm.IDAKLUSolver()
    exp = pybamm.Experiment(_EXPERIMENT_DESCRIPTIONS[experiment])
    return pybamm.Simulation(
        model, parameter_values=param, experiment=exp, solver=solver
    )


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("parameters", ["Marquis2019", "Chen2020"])
@pytest.mark.parametrize("experiment", ["CCCV", "GITT"])
def test_simulation_setup(benchmark, experiment, parameters, model_class):
    benchmark(_setup_sim, experiment, parameters, model_class)


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("parameters", ["Marquis2019", "Chen2020"])
@pytest.mark.parametrize("experiment", ["CCCV", "GITT"])
def test_simulation_solve(benchmark, experiment, parameters, model_class):
    sim = _setup_sim(experiment, parameters, model_class)
    benchmark(sim.solve)
