#
# Test leading surface form stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestLeadingOrderModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.Scalar(0)
        a_n = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["negative electrode"])
        a_s = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["separator"])
        a_p = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["positive electrode"])
        variables = {
            "Current collector current density": a,
            "Negative electrode porosity": a_n,
            "Negative electrolyte concentration": a_n,
            "Negative electrode interfacial current density": a_n,
            "X-averaged negative electrode interfacial current density": a,
            "X-averaged negative electrode total interfacial current density": a,
        }
        icd = " interfacial current density"
        reactions = {
            "main": {
                "Negative": {"s": 1, "aj": "Negative electrode" + icd},
                "Positive": {"s": 1, "aj": "Positive electrode" + icd},
            }
        }
        spf = pybamm.electrolyte_conductivity.surface_potential_form
        submodel = spf.LeadingOrderAlgebraic(param, "Negative", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.LeadingOrderDifferential(param, "Negative", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {
            "Current collector current density": a,
            "Negative electrolyte potential": a_n,
            "Negative electrolyte current density": a_n,
            "Separator electrolyte potential": a_s,
            "Separator electrolyte current density": a_s,
            "Positive electrode porosity": a_p,
            "Positive electrolyte concentration": a_p,
            "X-averaged positive electrode interfacial current density": a,
            "X-averaged positive electrode total interfacial current density": a,
        }
        submodel = spf.LeadingOrderAlgebraic(param, "Positive", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.LeadingOrderDifferential(param, "Positive", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
