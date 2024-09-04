#
# Tests for the lithium-ion SPM model
#
import pybamm
from tests import BaseIntegrationTestLithiumIon
import pytest


class TestSPM(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPM
