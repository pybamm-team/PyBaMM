#
# Test base particle submodel
#

import pybamm
import tests
import unittest


class TestCrackPropagation(unittest.TestCase):
    def test_public_functions(self):
        a_n = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["negative electrode"], "current collector"
        )
        a_p = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["positive electrode"], "current collector"
        )
        a = pybamm.Scalar(0)
        variables = {
            "Negative particle crack length": a_n,
            "Negative particle surface concentration": a_n,
            "R-averaged negative particle concentration": a_n,
            "Average negative particle concentration": a,
            "X-averaged cell temperature": a,
            "Negative electrode temperature": a_n,
            "Positive particle crack length": a_p,
            "Positive particle surface concentration": a_p,
            "R-averaged positive particle concentration": a_p,
            "Average positive particle concentration": a,
            "Positive electrode temperature": a_p,
            "Negative electrode active material volume fraction": a_n,
            "Negative electrode surface area to volume ratio": a_n,
            "Positive electrode active material volume fraction": a_p,
            "Positive electrode surface area to volume ratio": a_p,
        }
        options = {"particle": "Fickian diffusion", "particle cracking": "both"}
        param = pybamm.LithiumIonParameters(options)
        submodel = pybamm.particle_cracking.CrackPropagation(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = pybamm.particle_cracking.CrackPropagation(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()
        submodel = pybamm.particle_cracking.NoCracking(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
