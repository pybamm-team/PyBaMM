#
# Test constant concentration submodel
#

import pybamm
import tests
import unittest


class TestConstantConcentration(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.Scalar(0)
        variables = {"Porosity": a, "Electrolyte concentration": a}
        submodel = pybamm.electrolyte_diffusion.ConstantConcentration(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
