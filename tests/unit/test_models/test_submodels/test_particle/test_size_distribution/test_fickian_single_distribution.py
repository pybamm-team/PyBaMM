#
# Test single size distribution of fickian particles
#

import pybamm
import tests
import unittest


class TestSingleSizeDistribution(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()

        a = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "current collector")
        variables = {
            "X-averaged negative electrode interfacial current density distribution": a,
            "X-averaged negative electrode temperature": a,
            "Negative electrode active material volume fraction": a,
            "Negative electrode surface area to volume ratio": a,
        }

        submodel = pybamm.particle.FickianSingleSizeDistribution(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {
            "X-averaged positive electrode interfacial current density distribution": a,
            "X-averaged positive electrode temperature": a,
            "Positive electrode active material volume fraction": a,
            "Positive electrode surface area to volume ratio": a,
        }
        submodel = pybamm.particle.FickianSingleSizeDistribution(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
