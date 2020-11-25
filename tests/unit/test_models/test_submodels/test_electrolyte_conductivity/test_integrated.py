#
# Test integrated Stefan Maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.PrimaryBroadcast(1, "current collector")
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_s = pybamm.standard_variables.c_e_s
        c_e_p = pybamm.standard_variables.c_e_p
        variables = {
            "X-averaged electrolyte concentration": a,
            "Leading-order current collector current density": a,
            "Negative electrolyte concentration": c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Positive electrolyte concentration": c_e_p,
            "X-averaged negative electrode surface potential difference": a,
            "X-averaged negative electrode potential": a,
            "Negative electrolyte tortuosity": a,
            "Separator tortuosity": a,
            "Positive electrolyte tortuosity": a,
            "X-averaged cell temperature": a,
        }
        submodel = pybamm.electrolyte_conductivity.Integrated(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
