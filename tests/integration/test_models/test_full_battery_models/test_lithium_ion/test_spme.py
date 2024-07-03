#
# Tests for the lithium-ion SPMe model
#
import pybamm
from tests import BaseIntegrationTestLithiumIon
import pytest


class TestSPMe(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.run_basic_processing_test(options)
