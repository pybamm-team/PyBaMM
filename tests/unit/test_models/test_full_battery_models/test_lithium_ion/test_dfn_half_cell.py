#
# Tests for the lithium-ion half-cell DFN model
#

import pybamm
from tests import BaseUnitTestLithiumIonHalfCell
import pytest


class TestDFNHalfCell(BaseUnitTestLithiumIonHalfCell):
    def setup_method(self):
        self.model = pybamm.lithium_ion.DFN
