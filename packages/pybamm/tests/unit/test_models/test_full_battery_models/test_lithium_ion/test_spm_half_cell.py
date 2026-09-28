#
# Tests for the lithium-ion half-cell SPM model
#
import pybamm
from tests import BaseUnitTestLithiumIonHalfCell


class TestSPMHalfCell(BaseUnitTestLithiumIonHalfCell):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPM

    def test_well_posed_kinetics_asymmetric_butler_volmer(self):
        # SPM's own model-specific default sets "surface form" to "algebraic"
        # whenever "intercalation kinetics" is supplied.
        options = {
            "intercalation kinetics": "asymmetric Butler-Volmer",
            "surface form": "algebraic",
        }
        self.check_well_posedness(options)

    def test_well_posed_kinetics_linear(self):
        options = {"intercalation kinetics": "linear", "surface form": "algebraic"}
        self.check_well_posedness(options)

    def test_well_posed_kinetics_marcus(self):
        options = {"intercalation kinetics": "Marcus", "surface form": "algebraic"}
        self.check_well_posedness(options)

    def test_well_posed_kinetics_mhc(self):
        options = {
            "intercalation kinetics": "Marcus-Hush-Chidsey",
            "surface form": "algebraic",
        }
        self.check_well_posedness(options)
