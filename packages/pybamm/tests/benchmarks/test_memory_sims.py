import pytest

import pybamm

pytestmark = pytest.mark.memory_bench

_EXPERIMENT_DESCRIPTIONS = {
    "CCCV": [
        "Discharge at C/5 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 10 mA",
        "Rest for 1 hour",
    ],
    "GITT": [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20,
}

_MODELS = [
    pytest.param(pybamm.lithium_ion.SPM, id="spm"),
    pytest.param(pybamm.lithium_ion.DFN, id="dfn"),
]


@pytest.mark.limit_memory("1.5 MB")
@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("parameters", ["Marquis2019", "Chen2020"])
@pytest.mark.parametrize("experiment", ["CCCV", "GITT"])
def test_simulation_setup_memory(experiment, parameters, model_class):
    param = pybamm.ParameterValues(parameters)
    model = model_class()
    exp = pybamm.Experiment(_EXPERIMENT_DESCRIPTIONS[experiment])
    pybamm.Simulation(model, parameter_values=param, experiment=exp)
