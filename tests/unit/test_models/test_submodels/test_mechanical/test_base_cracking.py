#
# Test base cracking submodel
#

import pybamm
import tests
import unittest


class TestBaseCracking(unittest.TestCase):
    def test_public_functions(self):
        variables = {
            "Negative particle crack length": pybamm.Scalar(0),
            "Negative particle concentration": pybamm.Scalar(0),
            "Positive particle crack length": pybamm.Scalar(0),
            "Positive particle concentration": pybamm.Scalar(0),
        }
        submodel = pybamm.particle_cracking.BaseCracking(None, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
