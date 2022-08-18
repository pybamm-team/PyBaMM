#
# Tests for the polynomial profile submodel
#
import pybamm
import unittest


class TestParticlePolynomialProfile(unittest.TestCase):
    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "Particle type must be"):
            pybamm.particle.PolynomialProfile(None, "Negative", {})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
