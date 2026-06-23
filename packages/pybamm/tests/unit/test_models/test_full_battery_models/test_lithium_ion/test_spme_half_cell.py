"""Tests for the lithium-ion half-cell SPMe model (working electrode: positive)."""

import pybamm
from tests import BaseUnitTestLithiumIonHalfCell


class TestSPMeHalfCell(BaseUnitTestLithiumIonHalfCell):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPMe
