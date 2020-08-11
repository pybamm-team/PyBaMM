#
# Tests for the basic lithium-ion models
#
import pybamm
import unittest


class TestBasicModels(unittest.TestCase):
    def test_dfn_well_posed(self):
        model = pybamm.lithium_ion.BasicDFN()
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_spm_well_posed(self):
        model = pybamm.lithium_ion.BasicSPM()
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_dfn_half_cell_well_posed(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

        options = {"working electrode": "negative"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_dfn_half_cell_simulation_error(self):
        options = {"working electrode": "negative"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        with self.assertRaisesRegex(
            NotImplementedError, "not compatible with Simulations yet."
        ):
            pybamm.Simulation(model)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
