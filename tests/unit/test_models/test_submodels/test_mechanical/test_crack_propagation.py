#
# Test base particle submodel
#

import pybamm
import tests
import unittest


class TestBaseParticle(unittest.TestCase):
    def test_public_functions(self):
        variables = {
            "Negative particle crack length": 0,
            "Negative particle concentration": 0,
            # "Positive particle surface concentration": 0,
        }
        submodel = pybamm.particle_cracking.CrackPropagation(None, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        # submodel = pybamm.particle.BaseParticle(None, "Positive")
        # std_tests = tests.StandardSubModelTests(submodel, variables)
        # std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
