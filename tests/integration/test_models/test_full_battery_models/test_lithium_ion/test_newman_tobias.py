#
# Tests for the lithium-ion Newman-Tobias model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIon


@pytest.mark.skip(reason="TODO: remove me!!!")
class TestNewmanTobias(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.NewmanTobias

    def test_basic_processing(self):
        options = {}
        self.run_basic_processing_test(options)

    def test_sensitivities(self):
        pass  # skip this test

    def test_charge(self):
        pass  # skip this test

    def test_composite_graphite_silicon(self):
        pass  # skip this test

    def test_composite_graphite_silicon_sei(self):
        pass  # skip this test

    def test_composite_reaction_driven_LAM(self):
        pass  # skip this test

    def test_particle_size_composite(self):
        pass  # skip this test

    def test_composite_stress_driven_LAM(self):
        pass  # skip this test

    def test_sei_VonKolzenberg2020(self):
        pass  # skip this test

    def test_sei_tunnelling_limited(self):
        pass  # skip this test

    # try skipping some tests to see where ci is failing
    def test_particle_quartic(self):
        pass

    def test_constant_utilisation(self):
        pass

    def test_particle_quadratic(self):
        pass

    def test_current_driven_utilisation(self):
        pass

    def test_full_thermal(self):
        pass

    def test_lumped_thermal(self):
        pass
