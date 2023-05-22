#
# Tests for the surface formulation
#
import pybamm
import numpy as np
import unittest
from tests import TestCase


class TestCompareOutputsTwoPhase(TestCase):
    def compare_outputs_two_phase_graphite_graphite(self, model_class):
        """
        Check that a two-phase graphite-graphite model gives the same results as a
        standard one-phase graphite model
        """
        # Standard model
        model = model_class()
        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve([0, 3600])

        # Two phase model
        model_two_phase = model_class({"particle phases": ("2", "1")})

        ratio = pybamm.InputParameter("ratio")
        parameter_values_two_phase = pybamm.ParameterValues("Chen2020")

        for parameter in [
            "Negative electrode OCP [V]",
            "Negative electrode OCP entropic change [V.K-1]",
            "Maximum concentration in negative electrode [mol.m-3]",
            "Initial concentration in negative electrode [mol.m-3]",
            "Negative particle radius [m]",
            "Negative electrode diffusivity [m2.s-1]",
            "Negative electrode exchange-current density [A.m-2]",
        ]:
            parameter_values_two_phase.update(
                {
                    f"Primary: {parameter}": parameter_values_two_phase[parameter],
                    f"Secondary: {parameter}": parameter_values_two_phase[parameter],
                },
                check_already_exists=False,
            )
            del parameter_values_two_phase[parameter]
        parameter_values_two_phase.update(
            {
                "Primary: Negative electrode active material volume fraction"
                "": parameter_values_two_phase[
                    "Negative electrode active material volume fraction"
                ]
                * ratio,
                "Secondary: Negative electrode active material volume "
                "fraction": parameter_values_two_phase[
                    "Negative electrode active material volume fraction"
                ]
                * (1 - ratio),
            },
            check_already_exists=False,
        )
        del parameter_values_two_phase[
            "Negative electrode active material volume fraction"
        ]

        sim = pybamm.Simulation(
            model_two_phase, parameter_values=parameter_values_two_phase
        )
        for x in [0.1, 0.3, 0.5]:
            sol_two_phase = sim.solve([0, 3600], inputs={"ratio": x})
            # Compare two phase model to standard model
            for variable in [
                "X-averaged negative electrode active material volume fraction",
                "X-averaged negative electrode volumetric "
                "interfacial current density [A.m-3]",
                "Voltage [V]",
            ]:
                np.testing.assert_allclose(
                    sol[variable].entries, sol_two_phase[variable].entries, rtol=1e-2
                )

            # Compare each phase in the two-phase model
            np.testing.assert_allclose(
                sol_two_phase[
                    "Negative primary particle concentration [mol.m-3]"
                ].entries,
                sol_two_phase[
                    "Negative secondary particle concentration [mol.m-3]"
                ].entries,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                sol_two_phase[
                    "Negative electrode primary volumetric "
                    "interfacial current density [A.m-3]"
                ].entries
                / x,
                sol_two_phase[
                    "Negative electrode secondary volumetric "
                    "interfacial current density [A.m-3]"
                ].entries
                / (1 - x),
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                sol_two_phase[
                    "Negative electrode primary active material volume fraction"
                ].entries
                / x,
                sol_two_phase[
                    "Negative electrode secondary active material volume fraction"
                ].entries
                / (1 - x),
                rtol=1e-6,
            )

    def test_compare_SPM_graphite_graphite(self):
        model_class = pybamm.lithium_ion.SPM
        self.compare_outputs_two_phase_graphite_graphite(model_class)

    def test_compare_SPMe_graphite_graphite(self):
        model_class = pybamm.lithium_ion.SPMe
        self.compare_outputs_two_phase_graphite_graphite(model_class)

    def test_compare_DFN_graphite_graphite(self):
        model_class = pybamm.lithium_ion.DFN
        self.compare_outputs_two_phase_graphite_graphite(model_class)

    def compare_outputs_two_phase_silicon_graphite(self, model_class):
        # Check that increasing silicon content has the expected effect
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
        }
        model = model_class(options)

        name = "Negative electrode active material volume fraction"
        param = pybamm.ParameterValues("Chen2020_composite")
        x = pybamm.InputParameter("x")
        param.update(
            {
                f"Primary: {name}": (1 - x) * 0.75,
                f"Secondary: {name}": x * 0.75,
                "Current function [A]": 5 / 2,
            }
        )

        sim = pybamm.Simulation(model, parameter_values=param)
        t_eval = np.linspace(0, 9000, 1000)
        sol1 = sim.solve(t_eval, inputs={"x": 0.01})
        sol2 = sim.solve(t_eval, inputs={"x": 0.1})

        # Starting values should be close
        for var in [
            "Voltage [V]",
            "Average negative primary particle concentration",
            "Average negative secondary particle concentration",
        ]:
            np.testing.assert_allclose(
                sol1[var].data[:20], sol2[var].data[:20], rtol=1e-2
            )

        # More silicon means longer sim
        self.assertLess(sol1["Time [s]"].data[-1], sol2["Time [s]"].data[-1])

    def test_compare_SPM_silicon_graphite(self):
        model_class = pybamm.lithium_ion.SPM
        self.compare_outputs_two_phase_silicon_graphite(model_class)

    def test_compare_SPMe_silicon_graphite(self):
        model_class = pybamm.lithium_ion.SPMe
        self.compare_outputs_two_phase_silicon_graphite(model_class)

    def test_compare_DFN_silicon_graphite(self):
        model_class = pybamm.lithium_ion.DFN
        self.compare_outputs_two_phase_silicon_graphite(model_class)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
