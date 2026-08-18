#
# Tests for the lithium-ion SPM model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIon


class TestSPM(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPM
