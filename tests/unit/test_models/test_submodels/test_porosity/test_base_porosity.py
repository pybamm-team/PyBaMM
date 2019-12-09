#
# Test base porosity submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        a = pybamm.Scalar(0)
        variables = {"Negative electrode porosity": a, "Positive electrode porosity": a}
        submodel = pybamm.porosity.BaseModel(None)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
