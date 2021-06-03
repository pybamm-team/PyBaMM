#
# Test leading surface form stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestLeadingOrderModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.Scalar(0)
        a_n = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["negative electrode"], "current collector"
        )
        a_s = pybamm.FullBroadcast(pybamm.Scalar(0), ["separator"], "current collector")
        a_p = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["positive electrode"], "current collector"
        )
        variables = {
            "Current collector current density": a,
            "Negative electrode porosity": a_n,
            "Negative electrolyte concentration": a_n,
            "Sum of x-averaged negative electrode interfacial current densities": a,
            "X-averaged negative electrode total interfacial current density": a,
        }
        spf = pybamm.electrolyte_conductivity.surface_potential_form
        submodel = spf.LeadingOrderAlgebraic(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.LeadingOrderDifferential(param, "Negative")
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
            "Sum of x-averaged positive electrode interfacial current densities": a,
            "X-averaged positive electrode total interfacial current density": a,
        }
        submodel = spf.LeadingOrderAlgebraic(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.LeadingOrderDifferential(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
