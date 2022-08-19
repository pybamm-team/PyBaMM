#
# Tests for the surface formulation
#
import pybamm
import numpy as np
import unittest


class TestCompareOutputsTwoPhase(unittest.TestCase):
    def compare_outputs_two_phase(self, model_class):
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
            "Negative electrode electrons in reaction",
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
                "X-averaged negative electrode interfacial current density",
                "X-averaged negative electrode volumetric interfacial current density",
                "Terminal voltage [V]",
            ]:
                np.testing.assert_array_almost_equal(
                    sol[variable].entries, sol_two_phase[variable].entries, decimal=2
                )

            # Compare each phase in the two-phase model
            np.testing.assert_array_almost_equal(
                sol_two_phase["Negative primary particle concentration"].entries,
                sol_two_phase["Negative secondary particle concentration"].entries,
                decimal=6,
            )
            np.testing.assert_array_almost_equal(
                sol_two_phase[
                    "Negative electrode primary interfacial current density"
                ].entries
                / x,
                sol_two_phase[
                    "Negative electrode secondary interfacial current density"
                ].entries
                / (1 - x),
                decimal=6,
            )
            np.testing.assert_array_almost_equal(
                sol_two_phase[
                    "Negative electrode primary active material volume fraction"
                ].entries
                / x,
                sol_two_phase[
                    "Negative electrode secondary active material volume fraction"
                ].entries
                / (1 - x),
                decimal=6,
            )

    def test_compare_SPM(self):
        model_class = pybamm.lithium_ion.SPM
        self.compare_outputs_two_phase(model_class)

    def test_compare_SPMe(self):
        model_class = pybamm.lithium_ion.SPMe
        self.compare_outputs_two_phase(model_class)

    def test_compare_DFN(self):
        model_class = pybamm.lithium_ion.DFN
        self.compare_outputs_two_phase(model_class)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
