#
# Tests for the lithium-ion Newman-Tobias model
#
import numpy as np
import pytest

import pybamm
from tests import BaseUnitTestLithiumIon


class TestNewmanTobias(BaseUnitTestLithiumIon):
    def setup_method(self):
        self.model = pybamm.lithium_ion.NewmanTobias

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with pytest.raises(pybamm.OptionError, match=r"electrolyte conductivity"):
            pybamm.lithium_ion.NewmanTobias(options)

    def test_reaction_is_not_uniform_under_load(self):
        # x-averaged particles, but the potentials are resolved in x
        sim = pybamm.Simulation(pybamm.lithium_ion.NewmanTobias(), C_rate=2)
        solution = sim.solve([0, 600])
        x_n = sim.mesh["negative electrode"].nodes
        a_j = solution[
            "Negative electrode volumetric interfacial current density [A.m-3]"
        ](600, x=x_n)
        assert np.max(a_j) / np.min(a_j) > 1.5

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_particle_phases(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_particle_phases_thermal(self):
        pass  # Skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_particle_phases_heat_of_mixing(self):
        pass  # skip this test

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
    def test_well_posed_kinetics_coupled_ion_electron_transfer_composite(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_sei_VonKolzenberg2020_model(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_sei_tunnelling_limited(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_composite_differential_surface_form(self):
        pass  # skip this test

    @pytest.mark.skip(reason="Test currently not implemented")
    def test_well_posed_composite_algebraic_surface_form(self):
        pass  # skip this test

    @pytest.mark.skip(
        reason="Newman-Tobias + MSMR + particle size distribution is not "
        "supported (pre-existing DomainError in MSMR Butler-Volmer kinetics)"
    )
    def test_well_posed_psd_msmr_thermal(self):
        pass
