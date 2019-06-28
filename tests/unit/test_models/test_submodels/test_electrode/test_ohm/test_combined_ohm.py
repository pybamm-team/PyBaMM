#
# Test combined ohm submodel
#

import pybamm
import tests
import unittest


class TestCombinedModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode porosity": a,
            "Positive electrode porosity": a,
            "Average positive electrode open circuit potential": a,
            "Average positive electrode reaction overpotential": a,
            "Average positive electrolyte potential": a,
        }
        submodel = pybamm.electrode.ohm.Combined(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = pybamm.electrode.ohm.Combined(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
