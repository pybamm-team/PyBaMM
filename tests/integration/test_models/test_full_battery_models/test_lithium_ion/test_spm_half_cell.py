#
# Tests for the half-cell lithium-ion SPM model
#
import pybamm
from tests import BaseIntegrationTestLithiumIonHalfCell
import pytest


class TestSPMHalfCell(BaseIntegrationTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM
