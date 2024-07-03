#
# Tests for the lithium-ion Newman-Tobias model
#
import pybamm
from tests import BaseIntegrationTestLithiumIon
import pytest


class TestNewmanTobias(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setUp(self):
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
