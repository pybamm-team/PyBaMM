#
# Test edge cases for initial SOC
#
import numpy as np
import pytest

import pybamm


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

    @pytest.mark.parametrize("initial_soc", [0.5, "3.9 V"])
    def test_dfn_sensitivities_match_finite_differences(self, initial_soc):
        # inputs that change the stoichiometries the initial SOC maps to
        inputs = {
            "Positive electrode active material volume fraction": 0.665,
            "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
        }
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update({name: "[input]" for name in inputs})
        parameter_values["Current function [A]"] = 5.0
        sim = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            parameter_values=parameter_values,
            var_pts={"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 8, "r_p": 8},
            solver=pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12),
        )
        t = np.linspace(0, 600, 11)
        solution = sim.solve(
            [0, 600],
            t_interp=t,
            inputs=inputs,
            initial_soc=initial_soc,
            calculate_sensitivities=True,
        )
        sensitivities = {
            name: solution["Voltage [V]"].sensitivities[name].ravel() for name in inputs
        }
        for name, value in inputs.items():
            h = 1e-4 * value
            plus, minus = (
                sim.solve(
                    [0, 600],
                    t_interp=t,
                    inputs={**inputs, name: value + sign * h},
                    initial_soc=initial_soc,
                )["Voltage [V]"](t)
                for sign in [1, -1]
            )
            finite_difference = (plus - minus) / (2 * h)
            np.testing.assert_allclose(
                sensitivities[name],
                finite_difference,
                rtol=1e-4,
                atol=1e-4 * np.max(np.abs(finite_difference)),
            )
