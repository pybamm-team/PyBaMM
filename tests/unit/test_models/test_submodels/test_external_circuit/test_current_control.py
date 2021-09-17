#
# Test current control submodel
#

import pybamm
import tests
import unittest


class TestCurrentControl(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        submodel = pybamm.external_circuit.CurrentControl(param)
        std_tests = tests.StandardSubModelTests(submodel)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
