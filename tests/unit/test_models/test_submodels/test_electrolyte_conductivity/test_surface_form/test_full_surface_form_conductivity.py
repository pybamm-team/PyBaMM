#
# Test full surface form stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
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
            "Current collector current density": pybamm.PrimaryBroadcast(
                a, "current collector"
            ),
            "Negative electrode porosity": a_n,
            "Negative electrolyte tortuosity": a_n,
            "Negative electrode tortuosity": a_n,
            "Negative surface area per unit volume distribution in x": a_n,
            "Negative electrolyte concentration": a_n,
            "Sum of negative electrode interfacial current densities": a_n,
            "Electrolyte potential": pybamm.Concatenation(a_n, a_s, a_p),
            "Negative electrode temperature": a_n,
            "Separator temperature": a_s,
            "Positive electrode temperature": a_p,
            "Negative electrode potential": a_n,
            "Positive electrode potential": a_p,
        }

        spf = pybamm.electrolyte_conductivity.surface_potential_form
        submodel = spf.FullAlgebraic(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.FullDifferential(param, "Negative")
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
            "Positive surface area per unit volume distribution in x": a_p,
            "Positive electrolyte concentration": a_p,
            "Sum of positive electrode interfacial current densities": a_p,
            "Positive electrode temperature": a_p,
            "Negative electrode potential": a_n,
            "Positive electrode potential": a_p,
        }
        submodel = spf.FullAlgebraic(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.FullDifferential(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
