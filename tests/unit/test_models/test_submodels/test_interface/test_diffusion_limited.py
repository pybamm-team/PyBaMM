#
# Test diffusion limited submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_function(self):
        param = pybamm.standard_parameters_lead_acid

        a_n = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["negative electrode"])
        a_s = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["separator"])
        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode potential": a_n,
            "Negative electrolyte potential": a_n,
            "Negative electrolyte concentration": a_n,
            "X-averaged positive electrode oxygen interfacial current density": a,
            "Separator tortuosity": a_s,
            "Separator oxygen concentration": a_s,
            "Leading-order negative electrode oxygen interfacial current density": a,
        }
        submodel = pybamm.interface.DiffusionLimited(
            param, "Negative", "lead-acid oxygen", "leading"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = pybamm.interface.DiffusionLimited(
            param, "Negative", "lead-acid oxygen", "composite"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = pybamm.interface.DiffusionLimited(
            param, "Negative", "lead-acid oxygen", "full"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
