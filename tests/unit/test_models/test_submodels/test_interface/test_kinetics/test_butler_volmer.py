#
# Test base butler volmer submodel
#

import pybamm
import tests
import unittest


class TestButlerVolmer(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a_n = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["negative electrode"])
        a_p = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["positive electrode"])
        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode potential": a,
            "Negative electrolyte potential": a,
            "Negative electrode open circuit potential": a,
            "Negative electrolyte concentration": a,
            "Negative particle surface concentration": a,
            "Negative electrode temperature": a,
        }
        submodel = pybamm.interface.ButlerVolmer(param, "Negative", "lithium-ion main")
        std_tests = tests.StandardSubModelTests(submodel, variables)

        std_tests.test_all()

        variables = {
            "Current collector current density": a,
            "Positive electrode potential": a_p,
            "Positive electrolyte potential": a_p,
            "Positive electrode open circuit potential": a_p,
            "Positive electrolyte concentration": a_p,
            "Positive particle surface concentration": a_p,
            "Negative electrode interfacial current density": a_n,
            "Negative electrode exchange current density": a_n,
            "Positive electrode temperature": a_p,
        }
        submodel = pybamm.interface.ButlerVolmer(param, "Positive", "lithium-ion main")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
