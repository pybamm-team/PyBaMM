#
# Tests for the lithium-ion Newman-Tobias model
#
from tests import TestCase
import pybamm
import unittest
from tests import BaseUnitTestLithiumIon


class TestNewmanTobias(BaseUnitTestLithiumIon, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.NewmanTobias

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.lithium_ion.NewmanTobias(options)

    def test_well_posed_particle_phases(self):
        pass  # skip this test

    def test_well_posed_particle_phases_sei(self):
        pass  # skip this test

    def test_well_posed_composite_kinetic_hysteresis(self):
        pass  # skip this test

    def test_well_posed_composite_diffusion_hysteresis(self):
        pass  # skip this test


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
