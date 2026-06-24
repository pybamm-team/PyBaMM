import pybamm
from tests import BaseUnitTestLithiumIonHalfCell


class TestSPMeHalfCell(BaseUnitTestLithiumIonHalfCell):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPMe
