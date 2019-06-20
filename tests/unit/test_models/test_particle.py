import pybamm
import unittest


class TestParticle(unittest.TestCase):
    def test_exceptions(self):
        param = pybamm.standard_parameters_lithium_ion

        with self.assertRaises(pybamm.DomainError):
            pybamm.particle.fickian.ManyParticles(param, "not a domain")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
