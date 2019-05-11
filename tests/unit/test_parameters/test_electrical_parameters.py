#
# Tests for the electrical parameters
#
import pybamm

import os
import unittest


class TestElectricalParameters(unittest.TestCase):
    def test_current_functions(self):
        # create current functions
        dimensional_current = (
            pybamm.electrical_parameters.dimensional_current_density_with_time
        )
        dimensionless_current = pybamm.electrical_parameters.current_with_time

        # process
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height [m]": 0.1,
                "Electrode depth [m]": 0.1,
                "Number of electrodes connected in parallel to make a cell [-]": 8,
                "Typical current [A]": 2,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
            }
        )
        dimensional_current_eval = parameter_values.process_symbol(dimensional_current)
        dimensionless_current_eval = parameter_values.process_symbol(
            dimensionless_current
        )
        self.assertAlmostEqual(
            dimensional_current_eval.evaluate(t=3), 2 / (8 * 0.1 * 0.1)
        )
        self.assertEqual(dimensionless_current_eval.evaluate(t=3), 1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
