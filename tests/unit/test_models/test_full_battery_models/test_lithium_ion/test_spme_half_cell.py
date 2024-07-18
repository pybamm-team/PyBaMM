#
# Tests for the lithium-ion half-cell SPMe model
# This is achieved by using the {"working electrode": "positive"} option
#
import pybamm
from tests import BaseUnitTestLithiumIonHalfCell
import pytest


class TestSPMeHalfCell(BaseUnitTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe
