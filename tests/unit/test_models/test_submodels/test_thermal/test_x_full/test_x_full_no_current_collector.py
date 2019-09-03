#
# Test x-full thermal submodel with no current collectors
#

import pybamm
import tests
import unittest

from tests.unit.test_models.test_submodels.test_thermal.coupled_variables import (
    coupled_variables,
)


class TestNoCurrentCollector(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        submodel = pybamm.thermal.x_full.NoCurrentCollector(param)
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
