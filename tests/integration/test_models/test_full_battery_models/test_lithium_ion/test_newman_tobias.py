#
# Tests for the lithium-ion Newman-Tobias model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIon


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

    def test_composite_stress_driven_LAM(self):
        pass  # skip this test

    def test_sei_VonKolzenberg2020(self):
        pass  # skip this test

    def test_sei_tunnelling_limited(self):
        pass  # skip this test
