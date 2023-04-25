#
# Tests for the electrical parameters
#
from tests import TestCase
import pybamm

import unittest


class TestElectricalParameters(TestCase):
    def test_current_functions(self):
        # create current functions
        param = pybamm.electrical_parameters
        current = param.current_with_time
        current_density = param.current_density_with_time

        # process
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height [m]": 0.1,
                "Electrode width [m]": 0.1,
                "Number of electrodes connected in parallel to make a cell": 8,
                "Current function [A]": 2,
            }
        )
        current_eval = parameter_values.process_symbol(current)
        current_density_eval = parameter_values.process_symbol(current_density)

        # check current
        self.assertEqual(current_eval.evaluate(t=3), 2)

        # check current density
        self.assertAlmostEqual(current_density_eval.evaluate(t=3), 2 / (8 * 0.1 * 0.1))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
