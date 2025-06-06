#
# Tests for the polynomial profile submodel
#
import pytest

import pybamm


class TestParticlePolynomialProfile:
    def test_errors(self):
        with pytest.raises(ValueError, match="Particle type must be"):
            pybamm.particle.PolynomialProfile(None, "negative", {})
