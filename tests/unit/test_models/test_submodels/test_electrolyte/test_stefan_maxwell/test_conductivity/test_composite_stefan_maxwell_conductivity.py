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
            "Leading-order current collector current density": a,
            "Electrolyte concentration": c_e,
            "X-averaged electrolyte concentration": a,
            "X-averaged negative electrode potential": a,
            "X-averaged negative electrode surface potential difference": a,
            "Leading-order x-averaged negative electrode porosity": a,
            "Leading-order x-averaged separator porosity": a,
            "Leading-order x-averaged positive electrode porosity": a,
            "Cell temperature": a,
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
