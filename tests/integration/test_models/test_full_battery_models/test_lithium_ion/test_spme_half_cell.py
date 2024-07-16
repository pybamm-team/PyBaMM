#
# Tests for the half-cell lithium-ion SPMe model
#
import pybamm
from tests import BaseIntegrationTestLithiumIonHalfCell
import pytest


class TestSPMeHalfCell(BaseIntegrationTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe
