#
# Test lead acid inverse butler volmer submodel
#

import pybamm
import tests
import unittest


class TestLeadAcid(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        a = pybamm.Scalar(0)
        a_n = pybamm.Broadcast(pybamm.Scalar(0), ["negative electrode"])
        a_p = pybamm.Broadcast(pybamm.Scalar(0), ["positive electrode"])
        variables = {
            "Current collector current density": a,
            "Negative electrode interfacial current density": a_n,
            "Negative electrode open circuit potential": a_n,
            "Negative electrolyte concentration": a_n,
        }
        submodel = pybamm.interface.lead_acid.InverseButlerVolmer(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)

        std_tests.test_all()

        variables = {
            "Current collector current density": a,
            "Negative electrode interfacial current density": a_n,
            "Negative electrode exchange current density": a_n,
            "Positive electrode open circuit potential": a_p,
            "Positive electrolyte concentration": a_p,
        }
        submodel = pybamm.interface.lead_acid.InverseButlerVolmer(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
