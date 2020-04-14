#
# Test base stefan maxwell diffusion submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        variables = {"Electrolyte concentration": pybamm.Scalar(0)}
        submodel = pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion(None)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
