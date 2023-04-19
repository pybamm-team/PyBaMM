#
# Tests for the polynomial profile submodel
#
from tests import TestCase
import pybamm
import unittest


class TestParticlePolynomialProfile(TestCase):
    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "Particle type must be"):
            pybamm.particle.PolynomialProfile(None, "negative", {})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
