#
# Test single fast particles
#

import pybamm
import tests
import unittest


class TestSingleParticle(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "current collector")
        variables = {"X-averaged negative electrode interfacial current density": a}

        submodel = pybamm.particle.fast.SingleParticle(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {"X-averaged positive electrode interfacial current density": a}
        submodel = pybamm.particle.fast.SingleParticle(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
