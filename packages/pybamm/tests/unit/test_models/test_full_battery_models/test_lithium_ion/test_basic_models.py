#
# Tests for the basic lithium-ion models
#
import pybamm


class TestBasicModels:
    def test_dfn_well_posed(self):
        model = pybamm.lithium_ion.BasicDFN()
        model.check_well_posedness()

    def test_spm_well_posed(self):
        model = pybamm.lithium_ion.BasicSPM()
        model.check_well_posedness()

    def test_dfn_half_cell_well_posed(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

    def test_dfn_composite_well_posed(self):
        model = pybamm.lithium_ion.BasicDFNComposite()
        model.check_well_posedness()

    def test_dfn_2d(self):
        model = pybamm.lithium_ion.BasicDFN2D()
        model.check_well_posedness()

    def test_dfn_2d_unstructured(self):
        model = pybamm.lithium_ion.BasicDFN2DUnstructured(element_type="quad")
        model.check_well_posedness()

    def test_dfn_3d_unstructured(self):
        model = pybamm.lithium_ion.BasicDFN3DUnstructured()
        model.check_well_posedness()
