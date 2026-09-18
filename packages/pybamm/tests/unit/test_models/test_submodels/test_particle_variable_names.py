#
# Tests for the particle submodel variable names
#
import pytest

import pybamm


class TestParticleVariableNames:
    @pytest.mark.parametrize("domain", ["negative", "positive"])
    @pytest.mark.parametrize("stat", ["Minimum", "Maximum"])
    def test_particle_surface_concentration_names(self, domain, stat):
        model = pybamm.lithium_ion.SPMe()
        name = f"{stat} {domain} particle surface concentration [mol.m-3]"
        assert name in model.variables
