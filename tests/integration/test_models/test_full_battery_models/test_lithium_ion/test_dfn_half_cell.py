#
# Tests for the lithium-ion DFN half-cell model
#
import pybamm
from tests import BaseIntegrationTestLithiumIonHalfCell
import pytest


class TestDFNHalfCell(BaseIntegrationTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN
