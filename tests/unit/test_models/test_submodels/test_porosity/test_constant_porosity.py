#
# Test constant porosity submodel
#

import pybamm
import tests
import unittest


class TestConstantPorosity(unittest.TestCase):
    def test_public_functions(self):
        options = {
            "sei": "ec reaction limited",
            "sei film resistance": "distributed",
            "sei porosity change": "true",
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        param = pybamm.LithiumIonParameters()

        options = {
            "SEI": "ec reaction limited",
            "SEI film resistance": "distributed",
            "SEI porosity change": "true",
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }

        submodel = pybamm.porosity.Constant(param, options)
        std_tests = tests.StandardSubModelTests(submodel)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
