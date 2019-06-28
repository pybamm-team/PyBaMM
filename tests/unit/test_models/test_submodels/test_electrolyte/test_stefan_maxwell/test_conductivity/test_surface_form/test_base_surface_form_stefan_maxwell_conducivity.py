# Test base leading surface form stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode porosity": a,
            "Negative electrolyte concentration": a,
            "Negative electrode surface potential difference": a,
        }

        spf = pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form
        submodel = spf.BaseModel(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {
            "Current collector current density": a,
            "Negative electrolyte potential": a,
            "Negative electrolyte current density": a,
            "Separator electrolyte potential": a,
            "Separator electrolyte current density": a,
            "Positive electrode porosity": a,
            "Positive electrolyte concentration": a,
        }
        submodel = spf.BaseLeadingOrderModel(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
