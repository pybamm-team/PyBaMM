#
# Test many fast particles
#

import pybamm
import tests
import unittest


class TestManyParticles(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a_n = pybamm.FullBroadcast(
            pybamm.Scalar(0), "negative electrode", {"secondary": "current collector"}
        )
        a_p = pybamm.FullBroadcast(
            pybamm.Scalar(0), "positive electrode", {"secondary": "current collector"}
        )

        variables = {"Negative electrode interfacial current density": a_n}

        submodel = pybamm.particle.fast.ManyParticles(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {"Positive electrode interfacial current density": a_p}
        submodel = pybamm.particle.fast.ManyParticles(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
