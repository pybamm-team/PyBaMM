#
# Tests for the half-cell lithium-ion SPMe model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIonHalfCell


class TestSPMeHalfCell(BaseIntegrationTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPMe
