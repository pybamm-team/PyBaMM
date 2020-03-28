#
# Test full surface form stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.Scalar(0)
        a_n = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["negative electrode"], "current collector"
        )
        a_s = pybamm.FullBroadcast(pybamm.Scalar(0), ["separator"], "current collector")
        a_p = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["positive electrode"], "current collector"
        )
        variables = {
            "Current collector current density": pybamm.PrimaryBroadcast(
                a, "current collector"
            ),
            "Negative electrode porosity": a_n,
            "Negative electrolyte tortuosity": a_n,
            "Negative electrode tortuosity": a_n,
            "Negative electrolyte concentration": a_n,
            "Negative electrode interfacial current density": a_n,
            "Electrolyte potential": pybamm.Concatenation(a_n, a_s, a_p),
            "Negative electrode temperature": a_n,
            "Separator temperature": a_s,
            "Positive electrode temperature": a_p,
            "Negative electrode potential": a_n,
            "Positive electrode potential": a_p,
        }
        icd = " interfacial current density"
        reactions = {
            "main": {
                "Negative": {"s": 1, "aj": "Negative electrode" + icd},
                "Positive": {"s": 1, "aj": "Positive electrode" + icd},
            }
        }

        spf = pybamm.electrolyte_conductivity.surface_potential_form
        submodel = spf.FullAlgebraic(param, "Negative", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.FullDifferential(param, "Negative", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {
            "Current collector current density": pybamm.PrimaryBroadcast(
                a, "current collector"
            ),
            "Negative electrolyte potential": a_n,
            "Negative electrolyte current density": a_n,
            "Separator electrolyte potential": a_s,
            "Separator electrolyte current density": a_s,
            "Positive electrode porosity": a_p,
            "Positive electrolyte tortuosity": a_p,
            "Positive electrode tortuosity": a_p,
            "Positive electrolyte concentration": a_p,
            "Positive electrode interfacial current density": a_p,
            "Positive electrode temperature": a_p,
            "Negative electrode potential": a_n,
            "Positive electrode potential": a_p,
        }
        submodel = spf.FullAlgebraic(param, "Positive", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.FullDifferential(param, "Positive", reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
