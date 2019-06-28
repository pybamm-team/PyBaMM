#
# Test full capacitance stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestFullCapacitanceModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode porosity": a,
            "Negative electrolyte concentration": a,
            "Negative electrode interfacial current density": a,
        }

        spf = pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form
        submodel = spf.FullCapacitanceModel(param, "Negative")
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
            "Positive electrode interfacial current density": a,
        }
        submodel = spf.FullCapacitanceModel(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
