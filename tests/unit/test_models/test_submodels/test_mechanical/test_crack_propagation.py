#
# Test base particle submodel
#

import pybamm
import tests
import unittest


class TestCrackPropagation(unittest.TestCase):
    def test_public_functions(self):
        variables = {
            "Negative particle crack length": pybamm.Scalar(0),
            "Negative particle concentration": pybamm.Scalar(0),
            "Negative particle surface concentration": pybamm.Scalar(0),
            "R-averaged negative particle concentration": pybamm.Scalar(0),
            "Average negative particle concentration": pybamm.Scalar(0),
            "X-averaged cell temperature": pybamm.Scalar(0),
            "Negative electrode temperature": pybamm.Scalar(0),
        }
        options = {"particle cracking": "both"}
        param = pybamm.LithiumIonParameters(options)
        submodel = pybamm.particle_cracking.CrackPropagation(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
