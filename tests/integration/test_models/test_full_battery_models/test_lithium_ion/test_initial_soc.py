#
# Test edge cases for initial SOC
#
import pybamm
import pytest


class TestInitialSOC:
    @pytest.mark.parametrize(
        "param",
        [
            "Ai2020",
            "Chen2020",
            "Ecker2015",
            "Marquis2019",
            "Mohtat2020",
            "OKane2022",
            "ORegan2022",
        ],
    )
    def test_interpolant_parameter_sets(self, param):
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues(param)
        sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
        sim.solve([0, 600], initial_soc=0.2)
        sim.solve([0, 600], initial_soc="3.7 V")
