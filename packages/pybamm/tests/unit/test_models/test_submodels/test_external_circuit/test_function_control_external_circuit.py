import numpy as np
import pytest

import pybamm


def voltage_with_temperature(variables):
    # The temperature has a domain, which the control function cannot know
    T = variables["X-averaged cell temperature [K]"]
    return variables["Voltage [V]"] - 3.9 + 1e-3 * (T - 298.15)


class TestFunctionControl:
    def test_function_reads_variable_with_domain(self):
        model = pybamm.lithium_ion.SPM({"operating mode": voltage_with_temperature})
        solution = pybamm.Simulation(model).solve([0, 60])
        np.testing.assert_allclose(solution["Voltage [V]"](60), 3.9, rtol=1e-6)

    def test_experiment_step_reads_variable_with_domain(self):
        step = pybamm.step.CustomStepImplicit(voltage_with_temperature, duration=60)
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(), experiment=pybamm.Experiment([step])
        )
        solution = sim.solve()
        np.testing.assert_allclose(solution["Voltage [V]"](60), 3.9, rtol=1e-6)

    def test_experiment_step_missing_variable(self):
        step = pybamm.step.CustomStepImplicit(
            lambda variables: variables["Not a variable [V]"], duration=60
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(), experiment=pybamm.Experiment([step])
        )
        with pytest.raises(
            pybamm.ModelError,
            match=r"Missing variable for submodel 'external circuit': "
            r"'Not a variable \[V\]'",
        ):
            sim.solve()
