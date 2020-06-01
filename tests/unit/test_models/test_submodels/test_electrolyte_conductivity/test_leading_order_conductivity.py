#
# Test leading stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestLeadingOrder(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "X-averaged negative electrode potential": a,
            "X-averaged negative electrode surface potential difference": a,
        }
        submodel = pybamm.electrolyte_conductivity.LeadingOrder(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
