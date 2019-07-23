#
# Test single fickian particles
#

import pybamm
import tests
import unittest


class TestSingleParticle(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a = pybamm.Scalar(0)
        variables = {
            "Average negative electrode interfacial current density": a,
            "Average negative electrode temperature": a,
        }

        submodel = pybamm.particle.fickian.SingleParticle(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {
            "Average positive electrode interfacial current density": a,
            "Average positive electrode temperature": a,
        }
        submodel = pybamm.particle.fickian.SingleParticle(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
