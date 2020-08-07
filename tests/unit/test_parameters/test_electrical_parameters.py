#
# Tests for the electrical parameters
#
import pybamm

import unittest


class TestElectricalParameters(unittest.TestCase):
    def test_current_functions(self):
        # create current functions
        param = pybamm.ElectricalParameters()
        dimensional_current = param.dimensional_current_with_time
        dimensional_current_density = param.dimensional_current_density_with_time
        dimensionless_current = param.current_with_time
        dimensionless_current_density = param.current_with_time

        # process
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height [m]": 0.1,
                "Electrode width [m]": 0.1,
                "Number of electrodes connected in parallel to make a cell": 8,
                "Typical current [A]": 2,
                "Typical timescale [s]": 60,
                "Current function [A]": 2,
            }
        )
        dimensional_current_eval = parameter_values.process_symbol(dimensional_current)
        dimensional_current_density_eval = parameter_values.process_symbol(
            dimensional_current_density
        )
        dimensionless_current_eval = parameter_values.process_symbol(
            dimensionless_current
        )
        dimensionless_current_density_eval = parameter_values.process_symbol(
            dimensionless_current_density
        )

        # check current
        self.assertEqual(dimensional_current_eval.evaluate(t=3), 2)
        self.assertEqual(dimensionless_current_eval.evaluate(t=3), 1)

        # check current density
        self.assertAlmostEqual(
            dimensional_current_density_eval.evaluate(t=3), 2 / (8 * 0.1 * 0.1)
        )
        self.assertAlmostEqual(dimensionless_current_density_eval.evaluate(t=3), 1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
