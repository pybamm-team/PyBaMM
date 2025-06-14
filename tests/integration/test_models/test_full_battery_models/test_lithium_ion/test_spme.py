#
# Tests for the lithium-ion SPMe model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIon


class TestSPMe(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPMe

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.run_basic_processing_test(options)
