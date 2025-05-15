#
# Tests for the lithium-ion SPMe model
#
import pybamm
from tests import BaseIntegrationTestLithiumIon
import pytest


class TestSPMe(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        pybamm.set_logging_level("DEBUG")
        self.model = pybamm.lithium_ion.SPMe
