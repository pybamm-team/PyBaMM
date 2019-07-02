#
# Test leading-order concentration submodel
#

import pybamm
import tests
import unittest


class TestLeadingOrder(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid
        reactions = {
            "main": {
                "neg": {
                    "s_plus": param.s_n,
                    "j": "Average negative electrode interfacial current density",
                },
                "pos": {
                    "s_plus": param.s_p,
                    "j": "Average positive electrode interfacial current density",
                },
            }
        }
        a = pybamm.Scalar(0)
        variables = {
            "Average negative electrode porosity": a,
            "Average separator porosity": a,
            "Average positive electrode porosity": a,
            "Average negative electrode porosity change": a,
            "Average separator porosity change": a,
            "Average positive electrode porosity change": a,
            "Average negative electrode interfacial current density": a,
            "Average positive electrode interfacial current density": a,
        }
        submodel = pybamm.electrolyte.stefan_maxwell.diffusion.LeadingOrder(
            param, reactions
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
