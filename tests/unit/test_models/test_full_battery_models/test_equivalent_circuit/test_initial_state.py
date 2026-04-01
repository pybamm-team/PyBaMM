#
# Tests for equivalent circuit model (ECM) initial states
#

import pytest

import pybamm


class TestSetInitialSOC:
    def test_initial_soc(self):
        parameter_values = pybamm.ParameterValues("ECM_Example")
        V_min = parameter_values["Lower voltage cut-off [V]"]
        V_max = parameter_values["Upper voltage cut-off [V]"]

        param_0 = parameter_values.set_initial_state(0, inplace=False)
        param_100 = parameter_values.set_initial_state(1, inplace=False)
        assert param_0["Initial SoC"] == 0
        assert param_100["Initial SoC"] == 1

        param_0 = parameter_values.set_initial_state(f"{V_min} V", inplace=False)
        param_100 = parameter_values.set_initial_state(f"{V_max} V", inplace=False)
        assert param_0["Initial SoC"] == pytest.approx(0)
        assert param_100["Initial SoC"] == pytest.approx(1)

    def test_error(self):
        parameter_values = pybamm.ParameterValues("ECM_Example")

        with pytest.raises(ValueError, match=r"Initial SOC should be between 0 and 1"):
            parameter_values.set_initial_state(2)

        with pytest.raises(ValueError, match=r"outside the voltage limits"):
            parameter_values.set_initial_state("1 V")

        with pytest.raises(ValueError, match=r"must be a float"):
            parameter_values.set_initial_state("5 A")
