#
# Test lead acid butler volmer submodel
#

import pybamm
import tests
import unittest


class TestLeadAcid(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode potential": a,
            "Negative electrolyte potential": a,
            "Negative electrode open circuit potential": a,
            "Negative electrolyte concentration": a,
        }
        submodel = pybamm.interface.butler_volmer.LeadAcid(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)

        std_tests.test_all()

        variables = {
            "Current collector current density": a,
            "Positive electrode potential": a,
            "Positive electrolyte potential": a,
            "Positive electrode open circuit potential": a,
            "Positive electrolyte concentration": a,
            "Negative electrode interfacial current density": a,
            "Negative electrode exchange current density": a,
        }
        submodel = pybamm.interface.butler_volmer.LeadAcid(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
