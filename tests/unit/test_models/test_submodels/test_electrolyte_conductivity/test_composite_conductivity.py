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
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_s = pybamm.standard_variables.c_e_s
        c_e_p = pybamm.standard_variables.c_e_p
        variables = {
            "Leading-order current collector current density": a,
            "Negative electrolyte concentration": c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Positive electrolyte concentration": c_e_p,
            "X-averaged electrolyte concentration": a,
            "X-averaged negative electrode potential": a,
            "X-averaged negative electrode surface potential difference": a,
            "Leading-order x-averaged negative electrode porosity": a,
            "Leading-order x-averaged separator porosity": a,
            "Leading-order x-averaged positive electrode porosity": a,
            "Leading-order x-averaged negative electrolyte tortuosity": a,
            "Leading-order x-averaged separator tortuosity": a,
            "Leading-order x-averaged positive electrolyte tortuosity": a,
            "X-averaged cell temperature": a,
        }
        submodel = pybamm.electrolyte_conductivity.Composite(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

    def test_public_functions_first_order(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.PrimaryBroadcast(0, "current collector")
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_s = pybamm.standard_variables.c_e_s
        c_e_p = pybamm.standard_variables.c_e_p

        variables = {
            "Leading-order current collector current density": a,
            "Negative electrolyte concentration": c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Positive electrolyte concentration": c_e_p,
            "Leading-order x-averaged electrolyte concentration": a,
            "X-averaged electrolyte concentration": a,
            "X-averaged negative electrode potential": a,
            "X-averaged negative electrode surface potential difference": a,
            "Leading-order x-averaged negative electrode porosity": a,
            "Leading-order x-averaged separator porosity": a,
            "Leading-order x-averaged positive electrode porosity": a,
            "Leading-order x-averaged negative electrolyte tortuosity": a,
            "Leading-order x-averaged separator tortuosity": a,
            "Leading-order x-averaged positive electrolyte tortuosity": a,
            "X-averaged cell temperature": a,
        }
        submodel = pybamm.electrolyte_conductivity.Composite(
            param, higher_order_terms="first-order"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
