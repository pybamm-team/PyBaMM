#
# Test many fickian particles
#

import pybamm
import tests
import unittest


class TestManyParticles(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a = pybamm.Scalar(0)
        a_n = pybamm.Broadcast(0, ["negative electrode"])
        a_p = pybamm.Broadcast(0, ["positive electrode"])

        variables = {
            "Negative electrode interfacial current density": a_n,
            "Average negative electrode temperature": a,
        }

        submodel = pybamm.particle.fickian.ManyParticles(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {
            "Positive electrode interfacial current density": a_p,
            "Average positive electrode temperature": a,
        }
        submodel = pybamm.particle.fickian.ManyParticles(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
