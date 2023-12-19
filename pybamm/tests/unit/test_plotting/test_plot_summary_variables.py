import pybamm
import unittest
import numpy as np
from tests import TestCase


class TestPlotSummaryVariables(TestCase):
    def test_plot(self):
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at C/10 for 10 hours or until 3.3 V",
                    "Rest for 1 hour",
                    "Charge at 1 A until 4.1 V",
                    "Hold at 4.1 V until 50 mA",
                    "Rest for 1 hour",
                )
            ]
            * 3,
        )
        output_variables = [
            "Capacity [A.h]",
            "Loss of lithium inventory [%]",
            "Total capacity lost to side reactions [A.h]",
            "Loss of active material in negative electrode [%]",
            "Loss of active material in positive electrode [%]",
            "x_100",
            "x_0",
            "y_100",
            "y_0",
        ]
        sim = pybamm.Simulation(
            model, experiment=experiment, parameter_values=parameter_values
        )
        sol = sim.solve(initial_soc=1)

        axes = pybamm.plot_summary_variables(sol, testing=True)

        axes = axes.flatten()
        self.assertEqual(len(axes), 9)

        for output_var, ax in zip(output_variables, axes):
            self.assertEqual(ax.get_xlabel(), "Cycle number")
            self.assertEqual(ax.get_ylabel(), output_var)

            cycle_number, var = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(
                cycle_number, sol.summary_variables["Cycle number"]
            )
            np.testing.assert_array_equal(var, sol.summary_variables[output_var])

        axes = pybamm.plot_summary_variables(
            [sol, sol], labels=["SPM", "SPM"], testing=True
        )

        axes = axes.flatten()
        self.assertEqual(len(axes), 9)

        for output_var, ax in zip(output_variables, axes):
            self.assertEqual(ax.get_xlabel(), "Cycle number")
            self.assertEqual(ax.get_ylabel(), output_var)

            cycle_number, var = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(
                cycle_number, sol.summary_variables["Cycle number"]
            )
            np.testing.assert_array_equal(var, sol.summary_variables[output_var])

            cycle_number, var = ax.get_lines()[1].get_data()
            np.testing.assert_array_equal(
                cycle_number, sol.summary_variables["Cycle number"]
            )
            np.testing.assert_array_equal(var, sol.summary_variables[output_var])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
