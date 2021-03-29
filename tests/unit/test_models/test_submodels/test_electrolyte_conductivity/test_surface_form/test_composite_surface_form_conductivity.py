#
# Test leading surface form stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestCompositeModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.PrimaryBroadcast(1, "current collector")
        a_n = pybamm.FullBroadcast(
            pybamm.Scalar(1), ["negative electrode"], "current collector"
        )
        a_s = pybamm.FullBroadcast(pybamm.Scalar(1), ["separator"], "current collector")
        a_p = pybamm.FullBroadcast(
            pybamm.Scalar(1), ["positive electrode"], "current collector"
        )
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
            "Current collector current density": a,
            "Negative electrode porosity": a_n,
            "Sum of x-averaged negative electrode interfacial current densities": a,
            "X-averaged negative electrode total interfacial current density": a,
            "Current collector current density": a,
            "Negative electrolyte potential": a_n,
            "Negative electrolyte current density": a_n,
            "Separator electrolyte potential": a_s,
            "Separator electrolyte current density": a_s,
            "Positive electrode porosity": a_p,
            "Sum of x-averaged positive electrode interfacial current densities": a,
            "X-averaged positive electrode total interfacial current density": a,
        }

        spf = pybamm.electrolyte_conductivity.surface_potential_form
        submodel = spf.CompositeAlgebraic(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.CompositeDifferential(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = spf.CompositeAlgebraic(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = spf.CompositeDifferential(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
