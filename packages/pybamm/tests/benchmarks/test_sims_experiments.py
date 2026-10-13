import pytest

import pybamm

pytestmark = pytest.mark.time_bench

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
    exp = pybamm.Experiment(_EXPERIMENT_DESCRIPTIONS[experiment])
    return pybamm.Simulation(model, parameter_values=param, experiment=exp)


def _build_sim(experiment, parameters, model_class):
    sim = _setup_sim(experiment, parameters, model_class)
    sim.build_for_experiment()
    return sim


def _assert_completed(solution, experiment):
    # An infeasible step ends the experiment early, which would time as a speed-up
    assert len(solution.cycles) == len(_EXPERIMENT_DESCRIPTIONS[experiment])


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("parameters", ["Marquis2019", "Chen2020"])
@pytest.mark.parametrize("experiment", ["CCCV", "GITT"])
def test_simulation_build(benchmark, experiment, parameters, model_class):
    sim = benchmark(_build_sim, experiment, parameters, model_class)
    assert sim.steps_to_built_models


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("experiment", ["CCCV", "GITT"])
def test_simulation_first_solve(benchmark, experiment, model_class):
    # A freshly built simulation each round, so the solver set-up is timed
    def setup():
        return (_build_sim(experiment, "Chen2020", model_class),), {}

    solution = benchmark.pedantic(lambda sim: sim.solve(), setup=setup, rounds=5)
    _assert_completed(solution, experiment)


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("parameters", ["Marquis2019", "Chen2020"])
@pytest.mark.parametrize("experiment", ["CCCV", "GITT"])
def test_simulation_solve(benchmark, experiment, parameters, model_class):
    sim = _setup_sim(experiment, parameters, model_class)
    solution = benchmark(sim.solve)
    _assert_completed(solution, experiment)
