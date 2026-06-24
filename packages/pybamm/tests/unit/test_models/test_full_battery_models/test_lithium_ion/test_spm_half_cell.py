import pybamm
from tests import BaseUnitTestLithiumIonHalfCell


class TestSPMHalfCell(BaseUnitTestLithiumIonHalfCell):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPM
