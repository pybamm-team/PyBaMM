#
# Test x-lumped submodel with 2D current collectors
#

import pybamm
import tests
import unittest

from tests.unit.test_models.test_submodels.test_thermal.coupled_variables import (
    coupled_variables,
)


class TestCurrentCollector2D(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        phi_s_cn = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["current collector"])
        phi_s_cp = pybamm.PrimaryBroadcast(pybamm.Scalar(3), ["current collector"])

        coupled_variables.update(
            {
                "Negative current collector potential": phi_s_cn,
                "Positive current collector potential": phi_s_cp,
            }
        )

        submodel = pybamm.thermal.x_lumped.CurrentCollector2D(param)
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
