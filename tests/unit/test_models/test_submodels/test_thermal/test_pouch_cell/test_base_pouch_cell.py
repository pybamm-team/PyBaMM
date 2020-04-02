#
# Test base "N+1D" thermal submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        submodel = pybamm.thermal.pouch_cell.BasePouchCell(param, cc_dimension=1)
        std_tests = tests.StandardSubModelTests(submodel)
        with self.assertRaises(NotImplementedError):
            std_tests.test_all()

        submodel = pybamm.thermal.pouch_cell.BasePouchCell(param, cc_dimension=2)
        std_tests = tests.StandardSubModelTests(submodel)
        with self.assertRaises(NotImplementedError):
            std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
