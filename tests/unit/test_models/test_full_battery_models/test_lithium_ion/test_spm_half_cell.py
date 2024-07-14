#
# Tests for the lithium-ion half-cell SPM model
#
import pybamm
from tests import BaseUnitTestLithiumIonHalfCell
import pytest


class TestSPMHalfCell(BaseUnitTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM
