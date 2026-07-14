#
# Tests for the lithium-ion DFN half-cell model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIonHalfCell


class TestDFNHalfCell(BaseIntegrationTestLithiumIonHalfCell):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.DFN
