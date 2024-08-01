#
# Tests for the lithium-ion half-cell DFN model
#

import pybamm
from tests import BaseUnitTestLithiumIonHalfCell
import pytest


class TestDFNHalfCell(BaseUnitTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN
