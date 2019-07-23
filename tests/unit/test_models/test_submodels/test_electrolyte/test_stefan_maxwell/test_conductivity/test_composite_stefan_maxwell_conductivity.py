#
# Test combined stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestComposite(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.PrimaryBroadcast(0, "current collector")
        c_e = pybamm.standard_variables.c_e

        variables = {
            "Current collector current density": a,
            "Electrolyte concentration": c_e,
            "Average electrolyte concentration": a,
            "Average negative electrode potential": a,
            "Average negative electrode surface potential difference": a,
            "Average negative electrode porosity": a,
            "Average separator porosity": a,
            "Average positive electrode porosity": a,
            "Average cell temperature": a,
        }
        submodel = pybamm.electrolyte.stefan_maxwell.conductivity.Composite(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
