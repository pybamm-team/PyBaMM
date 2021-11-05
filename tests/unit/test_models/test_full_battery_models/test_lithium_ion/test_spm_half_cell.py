#
# Tests for the lithium-ion half-cell SPM model
# This is achieved by using the {"working electrode": "positive"} option
#
import pybamm
import unittest


class TestSPMHalfCell(unittest.TestCase):
    def test_well_posed(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


class TestSPMHalfCellWithSEI(unittest.TestCase):
    def test_well_posed_constant(self):
        options = {"working electrode": "positive", "SEI": "constant"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_reaction_limited(self):
        options = {"working electrode": "positive", "SEI": "reaction limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_solvent_diffusion_limited(self):
        options = {"working electrode": "positive", "SEI": "solvent-diffusion limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_electron_migration_limited(self):
        options = {"working electrode": "positive", "SEI": "electron-migration limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_interstitial_diffusion_limited(self):
        options = {
            "working electrode": "positive",
            "SEI": "interstitial-diffusion limited",
        }
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_ec_reaction_limited(self):
        options = {"working electrode": "positive", "SEI": "ec reaction limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
