#
# Test power control submodel
#

import pybamm
import tests
import unittest


class TestPowerControl(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        submodel = pybamm.external_circuit.PowerFunctionControl(param)
        variables = {"Terminal voltage [V]": pybamm.Scalar(0)}
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
