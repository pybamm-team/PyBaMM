#
# Tests for the lithium-ion Newman-Tobias model
#
import pytest

import pybamm
from tests import BaseUnitTestLithiumIon


class TestNewmanTobias(BaseUnitTestLithiumIon):
    def setup_method(self):
        self.model = pybamm.lithium_ion.NewmanTobias

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with pytest.raises(pybamm.OptionError, match="electrolyte conductivity"):
            pybamm.lithium_ion.NewmanTobias(options)

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_particle_phases(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_particle_phases_thermal(self):
        pass  # Skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_particle_phases_sei(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_composite_kinetic_hysteresis(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_composite_diffusion_hysteresis(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_composite_different_degradation(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_composite_LAM(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_sei_VonKolzenberg2020_model(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_sei_tunnelling_limited(self):
        pass  # skip this test
